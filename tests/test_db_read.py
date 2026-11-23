"""Read replica routing (app/db_read.py): replica when healthy, primary when it is down or
lagging, retry on the primary when the replica fails mid-query."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError

from app import db_read
from app.db_read import ReadRouter


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _engine(tmp_path: Path, name: str):
    engine = create_engine(f"sqlite:///{tmp_path / name}.db")
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE who (name TEXT)"))
        conn.execute(text("INSERT INTO who VALUES (:n)"), {"n": name})
    return engine


def _who(router: ReadRouter) -> str:
    with router.session() as s:
        return s.execute(text("SELECT name FROM who")).scalar_one()


@pytest.fixture
def lag(monkeypatch):
    """Set the replica's reported lag in seconds (the lag query is Postgres-only)."""

    def set_lag(seconds: float):
        monkeypatch.setattr(db_read, "LAG_SQL", text(f"SELECT {seconds}"))

    set_lag(0)
    return set_lag


def test_without_a_replica_reads_use_the_primary(tmp_path):
    router = ReadRouter(_engine(tmp_path, "primary"))
    assert _who(router) == "primary"
    assert router.status() == {"configured": False, "target": "primary"}


def test_healthy_replica_serves_reads(tmp_path, lag):
    router = ReadRouter(_engine(tmp_path, "primary"), _engine(tmp_path, "replica"))
    assert _who(router) == "replica"
    assert router.status()["target"] == "replica" and router.lag_seconds == 0


def test_unreachable_replica_falls_back_and_retries_later(tmp_path, lag):
    clock = Clock()
    down = create_engine(
        "postgresql+psycopg://u:p@127.0.0.1:1/none", connect_args={"connect_timeout": 1}
    )
    router = ReadRouter(_engine(tmp_path, "primary"), down, retry_seconds=30, clock=clock)
    attempts = []
    real_connect = down.connect
    down.connect = lambda: (attempts.append(1), real_connect())[1]  # type: ignore[method-assign]

    assert _who(router) == "primary"
    assert router.status()["target"] == "primary"
    assert "OperationalError" in router.status()["last_error"]
    tried = len(attempts)
    assert _who(router) == "primary" and len(attempts) == tried  # not retried within 30s

    clock.now += 31
    assert _who(router) == "primary" and len(attempts) > tried  # tried again, still down


def test_lagging_replica_is_skipped_until_it_catches_up(tmp_path, lag):
    clock = Clock()
    router = ReadRouter(
        _engine(tmp_path, "primary"),
        _engine(tmp_path, "replica"),
        max_lag_seconds=10,
        check_seconds=5,
        clock=clock,
    )
    lag(42)
    assert _who(router) == "primary" and router.lag_seconds == 42
    lag(0)
    assert _who(router) == "primary"  # lag is re-checked at most every 5s
    clock.now += 6
    assert _who(router) == "replica"


def _failing_on(router: ReadRouter, *, invalidated: bool, sqlstate: str | None = None):
    class Orig(Exception):
        pass

    orig = Orig("boom")
    orig.sqlstate = sqlstate  # type: ignore[attr-defined]
    calls: list[str] = []

    def fn(session):
        name = session.execute(text("SELECT name FROM who")).scalar_one()
        calls.append(name)
        if name == "replica":
            raise OperationalError("SELECT", {}, orig, connection_invalidated=invalidated)
        return name

    return fn, calls


def test_run_read_retries_on_the_primary_when_the_replica_connection_drops(tmp_path, lag):
    router = ReadRouter(_engine(tmp_path, "primary"), _engine(tmp_path, "replica"))
    fn, calls = _failing_on(router, invalidated=True)
    assert router.run(fn) == "primary"
    assert calls == ["replica", "primary"]
    assert router.status()["target"] == "primary"  # marked down for retry_seconds


def test_recovery_conflict_retries_on_the_primary_without_marking_the_replica_down(tmp_path, lag):
    router = ReadRouter(_engine(tmp_path, "primary"), _engine(tmp_path, "replica"))
    fn, calls = _failing_on(router, invalidated=False, sqlstate="40001")
    assert router.run(fn) == "primary"
    assert calls == ["replica", "primary"]
    assert router.status()["target"] == "replica"


def test_other_errors_are_not_retried(tmp_path, lag):
    router = ReadRouter(_engine(tmp_path, "primary"), _engine(tmp_path, "replica"))
    fn, calls = _failing_on(router, invalidated=False, sqlstate="42P01")
    with pytest.raises(OperationalError):
        router.run(fn)
    assert calls == ["replica"]


def test_configure_reads_ignores_a_missing_or_identical_url(tmp_path, monkeypatch):
    primary = _engine(tmp_path, "primary")
    monkeypatch.delenv("DB_READ_URL", raising=False)
    assert db_read.configure_reads(primary).replica is None
    assert db_read.configure_reads(primary, str(primary.url)).replica is None


def test_every_service_has_the_same_module():
    services = Path(__file__).resolve().parents[2]
    copies = {
        s: (services / s / "app" / "db_read.py").read_text()
        for s in ("api", "extraction", "work-items", "annotations")
    }
    assert len(set(copies.values())) == 1, "app/db_read.py differs between services"

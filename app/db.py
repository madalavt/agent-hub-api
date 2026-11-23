from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.pool import NullPool

from app.db_read import configure_reads


class Base(DeclarativeBase):
    pass


_engine: Engine | None = None
_SessionLocal: sessionmaker[Session] | None = None
_lock_engine: Engine | None = None


_STARTUP_LOCK_ID = 724111


@contextmanager
def startup_lock(engine: Engine) -> Iterator[None]:
    """One startup (schema, seeding) at a time across worker processes and replicas.
    Postgres advisory lock; no-op on SQLite."""
    if engine.dialect.name != "postgresql":
        yield
        return
    with engine.connect() as conn:
        conn.execute(text("SELECT pg_advisory_lock(:k)"), {"k": _STARTUP_LOCK_ID})
        try:
            yield
        finally:
            conn.execute(text("SELECT pg_advisory_unlock(:k)"), {"k": _STARTUP_LOCK_ID})


def init_db(database_url: str, direct_url: str | None = None) -> Engine:
    global _engine, _SessionLocal, _lock_engine
    if _engine is not None and _engine.url.render_as_string(hide_password=False) == database_url:
        return _engine
    kwargs: dict = {"pool_pre_ping": True}
    if database_url.startswith("sqlite"):
        path = database_url.split("///", 1)[-1]
        if path and path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        kwargs["connect_args"] = {"check_same_thread": False}
    if database_url.startswith("postgresql"):
        # Per worker process: workers × (DB_POOL_SIZE + DB_MAX_OVERFLOW) connections at most.
        kwargs.update(
            pool_size=int(os.environ.get("DB_POOL_SIZE", "5")),
            max_overflow=int(os.environ.get("DB_MAX_OVERFLOW", "5")),
            pool_timeout=10,
            pool_recycle=1800,
        )
    engine = create_engine(database_url, **kwargs)
    if database_url.startswith("sqlite"):

        @event.listens_for(engine, "connect")
        def _fk_on(dbapi_conn, _):  # pragma: no cover
            dbapi_conn.execute("PRAGMA foreign_keys=ON")

    from app import models  # noqa: F401

    schema = _direct_engine(database_url, direct_url) or engine
    with startup_lock(schema):
        Base.metadata.create_all(schema)
        _upgrade(schema)
    _lock_engine = schema if schema is not engine else None
    _engine = engine
    _SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
    configure_reads(engine)  # optional read replica (DB_READ_URL)
    return engine


PLATFORM_ID_SEQ = "projects_platform_id_seq"


def _upgrade(engine: Engine) -> None:
    """Idempotent schema upgrades for databases created by an older version (create_all only
    adds missing tables). Runs under the startup lock."""
    if engine.dialect.name != "postgresql":
        return
    with engine.begin() as conn:
        # Tenant roles (v1 roots-auth): a member's role per customer; last sign-in.
        conn.execute(
            text(
                "ALTER TABLE user_customers ADD COLUMN IF NOT EXISTS tenant_role_id VARCHAR(36)"
                " REFERENCES tenant_roles (id) ON DELETE SET NULL"
            )
        )
        conn.execute(
            text("ALTER TABLE user_customers ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ")
        )
        conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS last_active_at TIMESTAMPTZ"))
        # Tenant projects (v1 roots-auth projects): until now the UI showed tree configurations
        # as projects. The first time, every configuration becomes a project with the same id
        # (so links keep working); its queue is found by the queue's old project_id link until
        # queue_id is set (scripts/migrate_work_projects.py). Runs once (schema_migrations).
        conn.execute(
            text(
                "CREATE TABLE IF NOT EXISTS schema_migrations"
                " (name VARCHAR(100) PRIMARY KEY, applied_at TIMESTAMPTZ DEFAULT now())"
            )
        )
        done = conn.execute(
            text(
                "INSERT INTO schema_migrations (name) VALUES ('tenant_projects_from_configs')"
                " ON CONFLICT DO NOTHING RETURNING name"
            )
        ).first()
        if done:
            conn.execute(
                text(
                    "INSERT INTO tenant_projects (id, customer_id, name, description, created_at)"
                    " SELECT p.id, p.customer_id, p.name, p.description, p.created_at"
                    " FROM projects p WHERE NOT EXISTS"
                    " (SELECT 1 FROM tenant_projects t WHERE t.id = p.id)"
                )
            )
        # Decision trees belong to an org's library. Trees from before get the org of the
        # (oldest) configuration that uses them; unused ones stay admin-only until assigned.
        conn.execute(
            text(
                "ALTER TABLE trees ADD COLUMN IF NOT EXISTS customer_id VARCHAR(36)"
                " REFERENCES customers (id) ON DELETE SET NULL"
            )
        )
        conn.execute(
            text("CREATE INDEX IF NOT EXISTS ix_trees_customer_id ON trees (customer_id)")
        )
        trees_backfilled = conn.execute(
            text(
                "INSERT INTO schema_migrations (name) VALUES ('trees_customer_id_backfill')"
                " ON CONFLICT DO NOTHING RETURNING name"
            )
        ).first()
        if trees_backfilled:
            conn.execute(
                text(
                    "UPDATE trees t SET customer_id = u.customer_id FROM ("
                    " SELECT DISTINCT ON (tree_id) tree_id, customer_id FROM projects"
                    " WHERE tree_id IS NOT NULL ORDER BY tree_id, created_at, id) u"
                    " WHERE t.id = u.tree_id AND t.customer_id IS NULL"
                )
            )
        # projects.platform_id: integer project id (v1 style) for automation projects.
        conn.execute(text("ALTER TABLE projects ADD COLUMN IF NOT EXISTS platform_id INTEGER"))
        conn.execute(text(f"CREATE SEQUENCE IF NOT EXISTS {PLATFORM_ID_SEQ}"))
        # Imported v1 configurations keep their v1 project id (one project per id).
        conn.execute(
            text(
                "UPDATE projects p SET platform_id = c.v FROM ("
                " SELECT DISTINCT ON (v) id, v FROM ("
                "  SELECT id, created_at,"
                "  (parameters->>'doc_processing_project_id')::int AS v FROM projects"
                "  WHERE platform_id IS NULL"
                "  AND (parameters->>'doc_processing_project_id') ~ '^[0-9]{1,9}$') x"
                " WHERE NOT EXISTS (SELECT 1 FROM projects q WHERE q.platform_id = x.v)"
                " ORDER BY v, created_at, id) c WHERE p.id = c.id"
            )
        )
        conn.execute(
            text(
                "UPDATE projects p SET platform_id = s.base + s.rn FROM ("
                " SELECT id, ROW_NUMBER() OVER (ORDER BY created_at, id) AS rn,"
                " (SELECT COALESCE(MAX(platform_id), 0) FROM projects) AS base"
                " FROM projects WHERE platform_id IS NULL) s WHERE p.id = s.id"
            )
        )
        conn.execute(
            text(
                "SELECT CASE WHEN (SELECT COALESCE(MAX(platform_id), 0) FROM projects) > 0"
                f" THEN setval('{PLATFORM_ID_SEQ}', GREATEST("
                f"  (SELECT last_value FROM {PLATFORM_ID_SEQ}),"
                "  (SELECT MAX(platform_id) FROM projects)))"
                " END"
            )
        )
        conn.execute(
            text(
                "ALTER TABLE projects ALTER COLUMN platform_id"
                f" SET DEFAULT nextval('{PLATFORM_ID_SEQ}')"
            )
        )
        conn.execute(
            text(
                "CREATE UNIQUE INDEX IF NOT EXISTS uq_projects_platform_id"
                " ON projects (platform_id)"
            )
        )


def _direct_engine(database_url: str, direct_url: str | None) -> Engine | None:
    """A direct (unpooled) connection for startup work when the app connects through a
    transaction-pooling PgBouncer: session advisory locks and DDL need one server session."""
    if not direct_url or direct_url == database_url:
        return None
    return create_engine(direct_url, poolclass=NullPool, pool_pre_ping=True)


def lock_engine() -> Engine:
    """Engine for startup locks (seeding): the direct connection when there is one."""
    if _lock_engine is not None:
        return _lock_engine
    if _engine is None:
        raise RuntimeError("init_db() has not been called")
    return _engine


@contextmanager
def session_scope() -> Iterator[Session]:
    if _SessionLocal is None:
        raise RuntimeError("init_db() has not been called")
    session = _SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_session() -> Iterator[Session]:
    with session_scope() as session:
        yield session

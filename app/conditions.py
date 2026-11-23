"""Branch condition language.

A branch is followed when its condition is true for the fields extracted so far:

    (empty) / always                    -> always follow
    document_type == "Mortgage Change"  -> case-insensitive, whitespace-trimmed equality
    document_type == 'Loss Runs'        -> single quotes too (platform exports use them)
    document_type != "OTHERS"
    document_type in ["Check", "Claim"]
    document_type not in ["OTHERS"]
    policy_is_correct == "No" or claims_is_correct == "No"
    a == "x" and b != "y"               -> ``and`` binds tighter than ``or``; no parentheses

No ``eval``: conditions are tokenized and parsed here. The extraction service carries an
identical copy (services/extraction/app/engine/conditions.py); keep them in sync.
"""

from __future__ import annotations

import ast
from decimal import Decimal, InvalidOperation
import json
import re
from dataclasses import dataclass
from typing import Any

_CLAUSE_RE = re.compile(
    r"""^\s*(?P<key>[A-Za-z_][A-Za-z0-9_]*)\s*
        (?P<op>==|!=|not\s+in|in)\s*
        (?P<value>"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'|\[.*\]|-?\d+(?:\.\d+)?)\s*$""",
    re.VERBOSE | re.DOTALL,
)


class ConditionError(ValueError):
    pass


@dataclass(frozen=True)
class Clause:
    key: str
    op: str  # == != in not_in
    values: tuple[str, ...]


def _split(text: str, word: str) -> list[str]:
    # Split on a keyword surrounded by whitespace that is not inside quotes or brackets.
    parts, depth, quote, start, i = [], 0, "", 0, 0
    wlen = len(word)
    n = len(text)
    while i < n:
        ch = text[i]
        if quote:
            if ch == "\\" and i + 1 < n:
                i += 2
                continue
            if ch == quote:
                quote = ""
            i += 1
            continue
        if ch in "\"'":
            quote = ch
        elif ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
        elif depth == 0 and ch.isspace():
            j = i
            while j < n and text[j].isspace():
                j += 1
            if (
                j + wlen < n
                and text[j : j + wlen].lower() == word
                and text[j + wlen].isspace()
            ):
                k = j + wlen
                while k < n and text[k].isspace():
                    k += 1
                parts.append(text[start:i])
                start = k
                i = k
                continue
        i += 1
    parts.append(text[start:])
    return parts


def _norm(value: Any) -> str:
    return str(value).strip().lower()


def _as_decimal(value: str) -> Decimal | None:
    if not value:
        return None
    try:
        d = Decimal(value)
    except InvalidOperation:
        return None
    return d if d.is_finite() else None


def _equal(a: str, b: str) -> bool:
    if a == b:
        return True
    da, db = _as_decimal(a), _as_decimal(b)
    return da is not None and db is not None and da == db


def _parse_clause(text: str) -> Clause:
    m = _CLAUSE_RE.match(text)
    if not m:
        raise ConditionError(
            f'cannot parse {text.strip()!r}; expected e.g. document_type == "Mortgage Change"'
        )
    op = "not_in" if m["op"].startswith("not") else m["op"]
    raw = m["value"]
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        try:  # 'single quoted' strings and lists; literals only, never code
            value = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            raise ConditionError(f"invalid value {raw!r}: {exc}") from exc
    if op in ("in", "not_in"):
        if not isinstance(value, list):
            raise ConditionError(f'\'{m["op"]}\' needs a list, e.g. ["a", "b"]')
        values = tuple(_norm(v) for v in value)
    else:
        if isinstance(value, list):
            raise ConditionError(f"'{op}' needs a single value, not a list")
        values = (_norm(value),)
    return Clause(m["key"], op, values)


def parse(condition: str | None) -> list[list[Clause]]:
    """Return OR-groups of AND-ed clauses. ``[]`` means always true."""
    text = (condition or "").strip()
    if text.lower() in ("", "always", "true"):
        return []
    return [[_parse_clause(c) for c in _split(group, "and")] for group in _split(text, "or")]


def evaluate(condition: str | None, context: dict[str, Any]) -> bool:
    groups = parse(condition)
    if not groups:
        return True

    def check(clause: Clause) -> bool:
        actual = context.get(clause.key)
        actual_n = "" if actual is None else _norm(actual)
        if clause.op == "==":
            return _equal(actual_n, clause.values[0])
        if clause.op == "!=":
            return not _equal(actual_n, clause.values[0])
        if clause.op == "in":
            return any(_equal(actual_n, v) for v in clause.values)
        return not any(_equal(actual_n, v) for v in clause.values)

    return any(all(check(c) for c in group) for group in groups)

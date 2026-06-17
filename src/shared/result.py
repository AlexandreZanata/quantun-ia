"""Lightweight Result type — mirrors future TypeScript shared layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")
E = TypeVar("E")


@dataclass(frozen=True)
class Ok(Generic[T]):
    value: T


@dataclass(frozen=True)
class Fail(Generic[E]):
    error: E


Result = Ok[T] | Fail[E]


def ok(value: T) -> Ok[T]:
    return Ok(value=value)


def fail(error: E) -> Fail[E]:
    return Fail(error=error)

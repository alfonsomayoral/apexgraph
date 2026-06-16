"""Domain models for the task-tracker app.

Defines the two persisted entities — :class:`User` and :class:`Task` — plus a
couple of small helpers for turning database rows into model instances. These
are intentionally plain dataclasses: the persistence logic lives in
:mod:`db.pool`, keeping the models free of any I/O.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


def _utcnow() -> datetime:
    """Return a timezone-aware UTC timestamp (used as a default factory)."""
    return datetime.now(timezone.utc)


@dataclass
class User:
    """An application user.

    ``password_hash`` is never the plaintext password — :mod:`auth.service`
    hashes credentials before a ``User`` is ever constructed.
    """

    id: int
    email: str
    password_hash: str
    is_active: bool = True
    created_at: datetime = field(default_factory=_utcnow)

    @classmethod
    def from_row(cls, row: dict) -> "User":
        """Build a :class:`User` from a raw database row mapping."""
        return cls(
            id=row["id"],
            email=row["email"],
            password_hash=row["password_hash"],
            is_active=bool(row.get("is_active", True)),
            created_at=row.get("created_at", _utcnow()),
        )

    def public_dict(self) -> dict:
        """Serialize to a dict safe to return over the API (no password hash)."""
        return {"id": self.id, "email": self.email, "is_active": self.is_active}


@dataclass
class Task:
    """A unit of work owned by a :class:`User`."""

    id: int
    owner_id: int
    title: str
    done: bool = False
    created_at: datetime = field(default_factory=_utcnow)

    @classmethod
    def from_row(cls, row: dict) -> "Task":
        """Build a :class:`Task` from a raw database row mapping."""
        return cls(
            id=row["id"],
            owner_id=row["owner_id"],
            title=row["title"],
            done=bool(row.get("done", False)),
            created_at=row.get("created_at", _utcnow()),
        )

    def mark_done(self) -> None:
        """Mark this task as completed."""
        self.done = True

    def public_dict(self) -> dict:
        """Serialize to a dict suitable for an API response."""
        return {
            "id": self.id,
            "owner_id": self.owner_id,
            "title": self.title,
            "done": self.done,
        }

"""Server-side session storage.

A :class:`Session` ties an opaque token to a user id with an expiry. The
:class:`SessionStore` keeps live sessions in memory and is the single source of
truth consulted by :func:`auth.service.validate_token`. Tokens are random,
url-safe strings — they carry no user data themselves.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

# How long a freshly issued session stays valid.
DEFAULT_TTL = timedelta(hours=12)


@dataclass
class Session:
    """A single authenticated session."""

    token: str
    user_id: int
    expires_at: datetime

    def is_expired(self, now: datetime | None = None) -> bool:
        """Return ``True`` once the session has passed its expiry."""
        now = now or datetime.now(timezone.utc)
        return now >= self.expires_at


class SessionStore:
    """In-memory registry of live sessions keyed by token."""

    def __init__(self, ttl: timedelta = DEFAULT_TTL) -> None:
        self._ttl = ttl
        self._sessions: dict[str, Session] = {}

    def create_session(self, user_id: int) -> Session:
        """Mint a new session for ``user_id`` and store it."""
        token = secrets.token_urlsafe(32)
        expires_at = datetime.now(timezone.utc) + self._ttl
        session = Session(token=token, user_id=user_id, expires_at=expires_at)
        self._sessions[token] = session
        return session

    def get_session(self, token: str) -> Session | None:
        """Return the live session for ``token``, pruning it if expired."""
        session = self._sessions.get(token)
        if session is None:
            return None
        if session.is_expired():
            self.destroy_session(token)
            return None
        return session

    def destroy_session(self, token: str) -> bool:
        """Drop the session for ``token``; return whether one existed."""
        return self._sessions.pop(token, None) is not None

    def purge_expired(self) -> int:
        """Evict every expired session and return how many were removed."""
        now = datetime.now(timezone.utc)
        stale = [t for t, s in self._sessions.items() if s.is_expired(now)]
        for token in stale:
            self.destroy_session(token)
        return len(stale)

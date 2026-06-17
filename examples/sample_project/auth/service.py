"""Authentication service: credential checking and token lifecycle.

:class:`AuthService` is the orchestration layer for login. It verifies an
email/password pair against the user store (:mod:`db.pool`), and on success
asks the :class:`auth.session.SessionStore` to mint a session token. Logout and
token validation are thin wrappers over the same session store.
"""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass

from auth.session import Session, SessionStore
from db.models import User
from db.pool import ConnectionPool, find_user_by_email


def hash_password(password: str, *, salt: str = "apexgraph-demo") -> str:
    """Hash a plaintext password with a salted SHA-256 (demo-grade only)."""
    digest = hashlib.sha256()
    digest.update(salt.encode("utf-8"))
    digest.update(password.encode("utf-8"))
    return digest.hexdigest()


def verify_password(password: str, password_hash: str) -> bool:
    """Constant-time check that ``password`` hashes to ``password_hash``."""
    candidate = hash_password(password)
    return hmac.compare_digest(candidate, password_hash)


@dataclass
class LoginResult:
    """Outcome of a login attempt — either a token or a failure reason."""

    ok: bool
    token: str | None = None
    reason: str | None = None


class AuthService:
    """Coordinates credential verification and session issuance."""

    def __init__(self, pool: ConnectionPool, sessions: SessionStore) -> None:
        self._pool = pool
        self._sessions = sessions

    def login(self, email: str, password: str) -> LoginResult:
        """Authenticate a user and issue a session token on success.

        Looks the user up via :func:`db.pool.find_user_by_email`, checks the
        password with :func:`verify_password`, and mints a session through
        :meth:`auth.session.SessionStore.create_session`.
        """
        user = find_user_by_email(self._pool, email)
        if user is None:
            return LoginResult(ok=False, reason="no_such_user")
        if not self._check_user(user, password):
            return LoginResult(ok=False, reason="bad_credentials")
        session = self._sessions.create_session(user.id)
        return LoginResult(ok=True, token=session.token)

    def _check_user(self, user: User, password: str) -> bool:
        """Return whether ``user`` is active and the password matches."""
        if not user.is_active:
            return False
        return verify_password(password, user.password_hash)

    def logout(self, token: str) -> bool:
        """Invalidate ``token``; return whether a live session was destroyed."""
        return self._sessions.destroy_session(token)

    def validate_token(self, token: str) -> Session | None:
        """Resolve ``token`` to its live :class:`auth.session.Session`, if any."""
        return self._sessions.get_session(token)

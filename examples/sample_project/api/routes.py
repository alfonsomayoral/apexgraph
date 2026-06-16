"""HTTP-ish route handlers wiring auth and persistence together.

These handlers are framework-agnostic: each takes a simple ``request`` mapping
and returns a ``(status, body)`` tuple. They lean on :class:`auth.service.AuthService`
for authentication and on :mod:`db.pool` for reading and writing tasks. The
:func:`require_auth` helper turns a bearer token into a :class:`db.models.User`
context shared by the protected routes.
"""

from __future__ import annotations

from typing import Any

from auth.service import AuthService
from db.models import Task, User
from db.pool import (
    ConnectionPool,
    find_user_by_email,
    insert_task,
    list_tasks_for_user,
)
from auth.session import Session

Response = tuple[int, dict[str, Any]]


class Router:
    """Holds shared dependencies and exposes the route handlers."""

    def __init__(self, auth: AuthService, pool: ConnectionPool) -> None:
        self._auth = auth
        self._pool = pool

    # -- public routes ------------------------------------------------------

    def login_route(self, request: dict) -> Response:
        """POST /login — exchange credentials for a session token."""
        body = request.get("body", {})
        result = self._auth.login(body.get("email", ""), body.get("password", ""))
        if not result.ok:
            return 401, {"error": result.reason}
        return 200, {"token": result.token}

    def logout_route(self, request: dict) -> Response:
        """POST /logout — invalidate the caller's session token."""
        token = _bearer_token(request)
        destroyed = self._auth.logout(token) if token else False
        return 200, {"logged_out": destroyed}

    # -- protected routes ---------------------------------------------------

    def list_tasks_route(self, request: dict) -> Response:
        """GET /tasks — list the authenticated user's tasks."""
        session = self.require_auth(request)
        if session is None:
            return 401, {"error": "unauthorized"}
        tasks = list_tasks_for_user(self._pool, session.user_id)
        return 200, {"tasks": [t.public_dict() for t in tasks]}

    def create_task_route(self, request: dict) -> Response:
        """POST /tasks — create a task owned by the authenticated user."""
        session = self.require_auth(request)
        if session is None:
            return 401, {"error": "unauthorized"}
        title = request.get("body", {}).get("title", "").strip()
        if not title:
            return 400, {"error": "title_required"}
        task: Task = insert_task(self._pool, session.user_id, title)
        return 201, task.public_dict()

    def require_auth(self, request: dict) -> Session | None:
        """Resolve the bearer token on ``request`` to a live session."""
        token = _bearer_token(request)
        if not token:
            return None
        return self._auth.validate_token(token)

    def whoami_route(self, request: dict) -> Response:
        """GET /me — return the authenticated user's public profile."""
        session = self.require_auth(request)
        if session is None:
            return 401, {"error": "unauthorized"}
        user = _load_user(self._pool, session)
        if user is None:
            return 404, {"error": "user_gone"}
        return 200, user.public_dict()


def _bearer_token(request: dict) -> str | None:
    """Extract a bearer token from the ``Authorization`` header, if present."""
    header = request.get("headers", {}).get("Authorization", "")
    if header.startswith("Bearer "):
        return header[len("Bearer ") :]
    return None


def _load_user(pool: ConnectionPool, session: Session) -> User | None:
    """Best-effort reload of the user behind a session (demo lookup by id)."""
    # The demo store is keyed by email; a real app would look up by id.
    return find_user_by_email(pool, f"user{session.user_id}@example.com")

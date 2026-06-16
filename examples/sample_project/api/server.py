"""Application wiring — constructs the object graph and a route table.

:func:`build_app` is the composition root: it opens a :class:`db.pool.ConnectionPool`,
builds an :class:`auth.session.SessionStore` and :class:`auth.service.AuthService`
on top of it, hands both to the :class:`api.routes.Router`, and returns an
:class:`App` that maps ``(method, path)`` pairs to handlers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from auth.service import AuthService
from auth.session import SessionStore
from api.routes import Response, Router
from db.pool import ConnectionPool

Handler = Callable[[dict], Response]


@dataclass
class App:
    """A trivial dispatcher mapping routes to handlers."""

    router: Router
    routes: dict[tuple[str, str], Handler]

    def dispatch(self, method: str, path: str, request: dict) -> Response:
        """Route a request to its handler, or 404 if no route matches."""
        handler = self.routes.get((method.upper(), path))
        if handler is None:
            return 404, {"error": "not_found"}
        return handler(request)


def build_app(dsn: str = "file::memory:?cache=shared") -> App:
    """Compose the full application object graph and return a ready :class:`App`."""
    pool = ConnectionPool(dsn, size=4)
    sessions = SessionStore()
    auth = AuthService(pool, sessions)
    router = Router(auth, pool)

    routes: dict[tuple[str, str], Handler] = {
        ("POST", "/login"): router.login_route,
        ("POST", "/logout"): router.logout_route,
        ("GET", "/tasks"): router.list_tasks_route,
        ("POST", "/tasks"): router.create_task_route,
        ("GET", "/me"): router.whoami_route,
    }
    return App(router=router, routes=routes)


def main() -> None:
    """Build the app and print its route table (smoke test for wiring)."""
    app = build_app()
    for method, path in sorted(app.routes):
        print(f"{method:6} {path}")


if __name__ == "__main__":
    main()

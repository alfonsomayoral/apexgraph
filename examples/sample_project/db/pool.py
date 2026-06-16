"""A minimal connection pool and the data-access layer built on top of it.

:class:`ConnectionPool` hands out short-lived connections from a fixed-size
pool. The CRUD helpers (:func:`find_user_by_email`, :func:`insert_task`, ...)
take a pool, borrow a connection, run a query, and return model instances from
:mod:`db.models`. The rest of the app never touches raw connections directly.
"""

from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from typing import Iterator

from db.models import Task, User


class ConnectionPool:
    """A tiny thread-safe pool of SQLite connections.

    The pool eagerly opens ``size`` connections against ``dsn`` and lends them
    out via :meth:`acquire`. It is deliberately simple — enough to demonstrate
    borrow/return semantics without a real driver dependency.
    """

    def __init__(self, dsn: str, size: int = 5) -> None:
        self._dsn = dsn
        self._size = size
        self._lock = threading.Lock()
        self._free: list[sqlite3.Connection] = []
        self._open_all()

    def _open_all(self) -> None:
        """Eagerly create every connection the pool will manage."""
        for _ in range(self._size):
            conn = sqlite3.connect(self._dsn, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            self._free.append(conn)

    @contextmanager
    def acquire(self) -> Iterator[sqlite3.Connection]:
        """Borrow a connection for the duration of the ``with`` block."""
        conn = self._checkout()
        try:
            yield conn
        finally:
            self._checkin(conn)

    def _checkout(self) -> sqlite3.Connection:
        """Pop a free connection, raising if the pool is exhausted."""
        with self._lock:
            if not self._free:
                raise RuntimeError("connection pool exhausted")
            return self._free.pop()

    def _checkin(self, conn: sqlite3.Connection) -> None:
        """Return a borrowed connection to the free list."""
        with self._lock:
            self._free.append(conn)

    def close(self) -> None:
        """Close every pooled connection."""
        with self._lock:
            while self._free:
                self._free.pop().close()


def find_user_by_email(pool: ConnectionPool, email: str) -> User | None:
    """Look up a single active user by email, or ``None`` if absent."""
    with pool.acquire() as conn:
        cur = conn.execute(
            "SELECT * FROM users WHERE email = ? AND is_active = 1", (email,)
        )
        row = cur.fetchone()
    return User.from_row(dict(row)) if row else None


def insert_task(pool: ConnectionPool, owner_id: int, title: str) -> Task:
    """Insert a new task for ``owner_id`` and return the created :class:`Task`."""
    with pool.acquire() as conn:
        cur = conn.execute(
            "INSERT INTO tasks (owner_id, title, done) VALUES (?, ?, 0)",
            (owner_id, title),
        )
        conn.commit()
        task_id = cur.lastrowid
    return Task(id=task_id, owner_id=owner_id, title=title)


def list_tasks_for_user(pool: ConnectionPool, owner_id: int) -> list[Task]:
    """Return every task owned by ``owner_id``, newest first."""
    with pool.acquire() as conn:
        cur = conn.execute(
            "SELECT * FROM tasks WHERE owner_id = ? ORDER BY created_at DESC",
            (owner_id,),
        )
        rows = cur.fetchall()
    return [Task.from_row(dict(r)) for r in rows]

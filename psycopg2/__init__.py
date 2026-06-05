"""
psycopg2 compatibility shim — redirects to psycopg (v3).
Placed at the repo root so ml_pipeline imports resolve here when
psycopg2-binary is not installed (Python 3.14 has no wheel for it).
"""
import psycopg as _pg
from psycopg.rows import dict_row as _dict_row

# ── Connection ────────────────────────────────────────────────────────────────

class _Connection:
    """Thin wrapper that accepts psycopg2-style cursor_factory kwarg."""
    def __init__(self, conn):
        self._conn = conn

    def cursor(self, cursor_factory=None, name=None, **kwargs):
        from psycopg2.extras import RealDictCursor
        if cursor_factory is RealDictCursor:
            kwargs['row_factory'] = _dict_row
        return self._conn.cursor(**kwargs)

    def commit(self):   self._conn.commit()
    def rollback(self): self._conn.rollback()
    def close(self):    self._conn.close()

    def __enter__(self):        return self
    def __exit__(self, *a):     self._conn.__exit__(*a)
    def __getattr__(self, name): return getattr(self._conn, name)


def connect(*args, **kwargs):
    return _Connection(_pg.connect(*args, **kwargs))


# ── Exceptions (re-export) ────────────────────────────────────────────────────
Error             = _pg.Error
DatabaseError     = _pg.DatabaseError
IntegrityError    = _pg.IntegrityError
OperationalError  = _pg.OperationalError
ProgrammingError  = _pg.ProgrammingError
InterfaceError    = _pg.InterfaceError

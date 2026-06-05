"""
psycopg2.extras compatibility shim.
"""
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb


# ── Cursor factories ──────────────────────────────────────────────────────────

class RealDictCursor:
    """Sentinel recognised by psycopg2._Connection.cursor()."""
    pass


# ── JSON helpers ──────────────────────────────────────────────────────────────

def Json(obj):
    """psycopg2.extras.Json → psycopg Jsonb wrapper."""
    return Jsonb(obj)


# ── execute_values ────────────────────────────────────────────────────────────

def execute_values(cursor, sql, argslist, template=None, page_size=100, fetch=False):
    """
    psycopg2.extras.execute_values replacement.
    Automatically chunks rows so psycopg3's 65535-parameter limit is never hit.
    """
    argslist = list(argslist)
    if not argslist:
        return [] if fetch else None

    if template is None:
        n = len(argslist[0])
        template = '(' + ', '.join(['%s'] * n) + ')'

    # psycopg3 hard-limits parameters to 65535 per query
    params_per_row = template.count('%s')
    chunk_size = max(1, 65000 // params_per_row) if params_per_row > 0 else len(argslist)

    raw_cursor = getattr(cursor, '_cursor', cursor)
    results = []

    for i in range(0, len(argslist), chunk_size):
        chunk = argslist[i:i + chunk_size]
        placeholders = ', '.join([template] * len(chunk))
        flat_args = [v for row in chunk for v in row]
        full_sql = sql.replace('%s', placeholders, 1)
        raw_cursor.execute(full_sql, flat_args)
        if fetch:
            results.extend(raw_cursor.fetchall())

    return results if fetch else None

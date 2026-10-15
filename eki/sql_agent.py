"""SQL agent selects an approved query; no user or model generated SQL executes."""
import csv
import io
import sqlite3
from contextlib import closing
from decimal import Decimal, InvalidOperation


class SQLAgent:
    def __init__(self, path):
        self.path = path
        with closing(sqlite3.connect(path)) as db:
            db.execute("CREATE TABLE IF NOT EXISTS revenue (quarter TEXT, amount_cents INTEGER, source TEXT, source_row INTEGER, PRIMARY KEY(quarter,source))")
            db.commit()

    def import_csv(self, source, data):
        reader = csv.DictReader(io.StringIO(data.decode("utf-8-sig")))
        if set(reader.fieldnames or []) != {"quarter", "revenue"}:
            return False
        rows = []
        for n, row in enumerate(reader, 2):
            try:
                amount = Decimal(row['revenue'])
                if not amount.is_finite() or amount < 0 or amount > Decimal("1000000000000"):
                    raise ValueError("Invalid revenue amount")
                if amount * 100 != (amount * 100).to_integral_value():
                    raise ValueError("Revenue supports at most two decimal places")
                if row['quarter'] not in ("Q1", "Q2", "Q3", "Q4"):
                    raise ValueError("Quarter must be Q1–Q4")
                rows.append((row['quarter'], int(amount * 100), source, n))
            except (InvalidOperation, KeyError) as exc:
                raise ValueError("Invalid revenue CSV") from exc
        if len({r[0] for r in rows}) != len(rows):
            raise ValueError("Revenue CSV requires one row per quarter")
        with closing(sqlite3.connect(self.path)) as db:
            db.execute("DELETE FROM revenue WHERE source=?", (source,))
            db.executemany("INSERT INTO revenue VALUES (?,?,?,?)", rows)
            db.commit()
        return True

    def compare(self, first="Q1", second="Q2"):
        if first not in ("Q1", "Q2", "Q3", "Q4") or second not in ("Q1", "Q2", "Q3", "Q4"):
            raise ValueError("Unsupported quarter")
        uri = self.path.resolve().as_uri() + "?mode=ro"
        with closing(sqlite3.connect(uri, uri=True)) as db:
            db.execute("PRAGMA query_only=ON")
            db.set_authorizer(lambda action, *args: sqlite3.SQLITE_OK if action in
                              (sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION) else sqlite3.SQLITE_DENY)
            rows = db.execute("SELECT quarter, amount_cents, source, source_row FROM revenue WHERE quarter IN (?,?) ORDER BY quarter", (first, second)).fetchall()
        # Multiple uploaded datasets are ambiguous; never silently aggregate unrelated reports.
        if len(rows) != 2 or len({r[2] for r in rows}) != 1 or {r[0] for r in rows} != {first, second}:
            return None
        return {"first": first, "second": second, "rows": rows,
                "query_id": "revenue_quarter_comparison_v1", "parameters": [first, second]}


def analyze(result):
    values = {row[0]: row[1] for row in result['rows']}
    before, after = values[result['first']], values[result['second']]
    delta = after - before
    return {"before": before / 100, "after": after / 100, "delta": delta / 100,
            "change_percent": round(delta / before * 100, 2) if before else None,
            "baseline_zero": before == 0}

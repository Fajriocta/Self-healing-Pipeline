import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "database" / "sales.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS sales (
    transaction_id  TEXT PRIMARY KEY,
    customer_email  TEXT NOT NULL,
    purchase_amount REAL NOT NULL,
    purchase_date   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS healing_log (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ran_at       TEXT NOT NULL,
    source_file  TEXT NOT NULL,
    engine       TEXT NOT NULL,
    drift_found  INTEGER NOT NULL,
    mapping      TEXT NOT NULL,
    rows_loaded  INTEGER NOT NULL,
    rows_quarantined INTEGER NOT NULL
);
"""


def main() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript(SCHEMA)
    print(f"OK  database siap: {DB_PATH}")


if __name__ == "__main__":
    main()

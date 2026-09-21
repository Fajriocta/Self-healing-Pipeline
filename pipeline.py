from __future__ import annotations

import json
import logging
import re
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

import init_db
from healer import get_healer

ROOT = Path(__file__).parent

RAW_DIR = ROOT / "data" / "raw"
HEALED_DIR = ROOT / "data" / "healed"
OUTPUT_DIR = ROOT / "data" / "output"
QUARANTINE_DIR = ROOT / "data" / "quarantine"

REFERENCE_CSV = ROOT / "data" / "ref" / "customers.csv"
JOIN_KEY = "customer_email"

DB_PATH = ROOT / "database" / "sales.db"
TABLE = "sales"

ENGINE = "auto"
LOAD_TO_DB = True

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
log = logging.getLogger("pipeline")

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass


def expected_schema(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(f"PRAGMA table_info({TABLE})").fetchall()
    if not rows:
        raise SystemExit(f"Tabel {TABLE} tidak ada di {DB_PATH}.")
    return [r[1] for r in rows]


def align_schema(df: pd.DataFrame, expected: list[str], healer) -> tuple[pd.DataFrame, dict, str]:
    actual = list(df.columns)
    engine_name = getattr(healer, "label", healer.name)

    if not set(expected) - set(actual):
        return df[expected], {}, engine_name

    mapping = healer.heal(expected, actual)
    df = df.rename(columns=mapping)

    still_missing = [c for c in expected if c not in df.columns]
    if still_missing:
        raise RuntimeError(f"tidak ada padanan untuk {still_missing}")

    extra = [c for c in df.columns if c not in expected]
    if extra:
        log.info("  skip kolom %s", extra)

    return df[expected], mapping, engine_name


def validate(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    reasons = pd.Series("", index=df.index)

    def flag(mask: pd.Series, why: str) -> None:
        reasons[mask & (reasons == "")] = why

    flag(df["transaction_id"].isna() | (df["transaction_id"].astype(str).str.strip() == ""),
         "transaction_id kosong")

    email = df["customer_email"].astype(str).str.strip()
    flag(~email.str.match(EMAIL_RE), "email tidak valid")

    amount = pd.to_numeric(df["purchase_amount"], errors="coerce")
    flag(amount.isna(), "purchase_amount bukan angka")
    flag(amount.notna() & (amount <= 0), "purchase_amount <= 0")

    date = pd.to_datetime(df["purchase_date"], errors="coerce", format="mixed")
    flag(date.isna(), "purchase_date tidak bisa diparsing")

    df["purchase_amount"] = amount
    df["purchase_date"] = date.dt.strftime("%Y-%m-%d")
    df["customer_email"] = email

    bad = reasons != ""
    return df[~bad], df[bad].assign(_reason=reasons[bad])


def join_reference(df: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    joined = df.merge(ref, on=JOIN_KEY, how="left")
    penanda = ref.columns.drop(JOIN_KEY)[0]
    yatim = int(joined[penanda].isna().sum())
    if yatim:
        log.warning("  %d baris tanpa pasangan di %s", yatim, REFERENCE_CSV.name)
    return joined


def load(conn: sqlite3.Connection, df: pd.DataFrame, expected: list[str]) -> int:
    cols = ", ".join(expected)
    placeholders = ", ".join("?" * len(expected))
    cur = conn.executemany(
        f"INSERT OR IGNORE INTO {TABLE} ({cols}) VALUES ({placeholders})",
        df[expected].itertuples(index=False, name=None),
    )
    conn.commit()
    return cur.rowcount


def process_one(csv_path: Path, conn: sqlite3.Connection, expected: list[str],
                ref: pd.DataFrame, healer) -> dict:
    df = pd.read_csv(csv_path)
    log.info("%s (%d baris)", csv_path.name, len(df))

    df, mapping, engine_name = align_schema(df, expected, healer)

    df.to_csv(HEALED_DIR / csv_path.name, index=False)

    clean, dirty = validate(df)
    if not dirty.empty:
        dirty.to_csv(QUARANTINE_DIR / csv_path.name, index=False)
        for reason, n in dirty["_reason"].value_counts().items():
            log.warning("  drop %d baris: %s", n, reason)

    joined = join_reference(clean, ref)
    joined.to_csv(OUTPUT_DIR / csv_path.name, index=False)
    log.info("  out %d baris x %d kolom", len(joined), len(joined.columns))

    dimuat = 0
    if LOAD_TO_DB:
        dimuat = load(conn, clean, expected)
        conn.execute(
            "INSERT INTO healing_log (ran_at, source_file, engine, drift_found, mapping,"
            " rows_loaded, rows_quarantined) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (datetime.now().isoformat(timespec="seconds"), csv_path.name, engine_name,
             int(bool(mapping)), json.dumps(mapping), dimuat, len(dirty)),
        )
        conn.commit()

    return {"file": csv_path.name, "status": "OK", "heal": len(mapping),
            "join": len(joined), "buang": len(dirty), "db": dimuat}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not DB_PATH.exists():
        init_db.main()

    for folder in (HEALED_DIR, OUTPUT_DIR, QUARANTINE_DIR):
        folder.mkdir(parents=True, exist_ok=True)
        for sisa in folder.glob("*.csv"):
            sisa.unlink()

    ref = pd.read_csv(REFERENCE_CSV)
    sumber = sorted(RAW_DIR.glob("*.csv"))

    healer = get_healer(ENGINE)

    hasil = []
    with sqlite3.connect(DB_PATH) as conn:
        expected = expected_schema(conn)
        log.info("%d file | target: %s", len(sumber), ", ".join(expected))
        for csv_path in sumber:
            try:
                hasil.append(process_one(csv_path, conn, expected, ref, healer))
            except Exception as exc:
                log.error("  GAGAL: %s", exc)
                hasil.append({"file": csv_path.name, "status": "GAGAL", "heal": 0,
                              "join": 0, "buang": 0, "db": 0})

    log.info("%-24s %-6s %5s %5s %6s %4s", "FILE", "STATUS", "HEAL", "JOIN", "BUANG", "DB")
    for r in hasil:
        log.info("%-24s %-6s %5d %5d %6d %4d", r["file"], r["status"],
                 r["heal"], r["join"], r["buang"], r["db"])


if __name__ == "__main__":
    main()

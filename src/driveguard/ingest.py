"""DuckDB-based schema-aligned ingestion of Backblaze daily CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb


def ingest(raw_dir: Path, output: Path) -> None:
    files = list(raw_dir.rglob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found under {raw_dir}")
    output.parent.mkdir(parents=True, exist_ok=True)
    # DuckDB union_by_name tolerates quarter-to-quarter column additions. Keep
    # parse failures visible; ignore_errors would silently hide data problems.
    glob = str(raw_dir / "**" / "*.csv").replace("'", "''")
    out = str(output).replace("'", "''")
    con = duckdb.connect()
    try:
        con.execute(
            f"COPY (SELECT * FROM read_csv( '{glob}', header=true, union_by_name=true, "
            "filename=true, all_varchar=true, nullstr='') ) "
            f"TO '{out}' (FORMAT PARQUET, COMPRESSION ZSTD)"
        )
        count = con.execute(f"SELECT count(*) FROM read_parquet('{out}')").fetchone()[0]
        print(f"Wrote {count:,} rows from {len(files):,} files to {output}")
    finally:
        con.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/drive_days.parquet"))
    args = parser.parse_args()
    ingest(args.raw_dir, args.output)


if __name__ == "__main__":
    main()

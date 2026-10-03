"""Generate horizon labels in DuckDB without materializing the fleet in RAM."""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb


def build_labels(source: Path, output: Path, horizon_days: int = 30) -> None:
    if horizon_days < 1:
        raise ValueError("horizon_days must be positive")
    if not source.exists():
        raise FileNotFoundError(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    src = str(source).replace("'", "''")
    dest = str(output).replace("'", "''")
    con = duckdb.connect()
    try:
        con.execute(
            f"""
            COPY (
              WITH parsed AS (
                SELECT *, try_cast(date AS DATE) AS observation_date,
                       coalesce(try_cast(failure AS INTEGER), 0) AS failure_flag,
                       serial_number AS drive_id
                FROM read_parquet('{src}')
              ), events AS (
                SELECT drive_id,
                       min(observation_date) FILTER (WHERE failure_flag = 1) AS failure_date,
                       max(observation_date) AS last_seen
                FROM parsed GROUP BY drive_id
              ), joined AS (
                SELECT p.*, e.failure_date, e.last_seen,
                       date_diff('day', p.observation_date, e.failure_date) AS days_to_failure
                FROM parsed p JOIN events e USING (drive_id)
              ), labeled AS (
                SELECT *,
                  CASE
                    WHEN days_to_failure BETWEEN 1 AND {horizon_days} THEN 1
                    WHEN days_to_failure > {horizon_days} THEN 0
                    WHEN failure_date IS NULL
                         AND date_diff('day', observation_date, last_seen) >= {horizon_days} THEN 0
                    ELSE NULL
                  END AS will_fail_30d,
                  CASE
                    WHEN failure_flag = 1 THEN 'failure_day'
                    WHEN days_to_failure BETWEEN 1 AND {horizon_days} THEN 'positive'
                    WHEN days_to_failure > {horizon_days} THEN 'negative_before_later_failure'
                    WHEN failure_date IS NULL
                         AND date_diff('day', observation_date, last_seen) >= {horizon_days} THEN 'negative'
                    ELSE 'insufficient_followup'
                  END AS label_status
                FROM joined
              )
              SELECT * EXCLUDE (observation_date, failure_flag, drive_id, days_to_failure)
              FROM labeled WHERE failure_flag = 0
            ) TO '{dest}' (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
        report = con.execute(
            f"SELECT label_status, count(*) FROM read_parquet('{dest}') GROUP BY 1 ORDER BY 1"
        ).fetchall()
        print("Label counts:")
        for status, count in report:
            print(f"  {status}: {count:,}")
        print(f"Wrote labeled observations to {output}")
    finally:
        con.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/processed/drive_days.parquet"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/labeled.parquet"))
    parser.add_argument("--horizon-days", type=int, default=30)
    args = parser.parse_args()
    build_labels(args.input, args.output, args.horizon_days)


if __name__ == "__main__":
    main()

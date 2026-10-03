"""As-of drive-day features built with bounded DuckDB time windows."""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb


SMART_FIELDS = (5, 9, 187, 194, 197, 198)
CHANGE_FIELDS = (5, 187, 197, 198)
WINDOW_DAYS = (7, 30)


def build_features(source: Path, labels: Path, output: Path) -> None:
    for path in (source, labels):
        if not path.exists():
            raise FileNotFoundError(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    src, lab, dest = (str(p).replace("'", "''") for p in (source, labels, output))
    con = duckdb.connect()
    try:
        available = {
            row[0]
            for row in con.execute(
                "DESCRIBE SELECT * FROM read_parquet(?)", [str(source)]
            ).fetchall()
        }
        candidates = [n for n in SMART_FIELDS if f"smart_{n}_raw" in available]
        if candidates:
            coverage_sql = ", ".join(
                f"count(try_cast(smart_{n}_raw AS DOUBLE)) AS n{n}" for n in candidates
            )
            coverage = con.execute(
                f"SELECT {coverage_sql} FROM read_parquet('{src}')"
            ).fetchone()
            smart = [n for n, count in zip(candidates, coverage) if count]
            excluded = [n for n, count in zip(candidates, coverage) if not count]
        else:
            smart, excluded = [], []
        if not smart:
            raise ValueError("No selected raw SMART fields exist in the ingested schema")
        if excluded:
            print(f"Skipping fully missing SMART attributes: {excluded}")

        expressions = [
            "try_cast(capacity_bytes AS DOUBLE) AS capacity_bytes",
            "model",
        ]
        expressions.extend(
            f"try_cast(smart_{n}_raw AS DOUBLE) AS smart_{n}_raw_num" for n in smart
        )
        typed = ",\n                       ".join(expressions)

        rolling = []
        for n in smart:
            col = f"smart_{n}_raw_num"
            rolling.append(f"CASE WHEN {col} IS NULL THEN 1 ELSE 0 END AS smart_{n}_missing")
            for days in WINDOW_DAYS:
                frame = f"RANGE BETWEEN INTERVAL {days - 1} DAYS PRECEDING AND CURRENT ROW"
                for stat, agg in (("mean", "avg"), ("max", "max"), ("std", "stddev_pop"), ("count", "count")):
                    rolling.append(
                        f"{agg}({col}) OVER (PARTITION BY serial_number ORDER BY obs_date {frame}) "
                        f"AS smart_{n}_{stat}_{days}d"
                    )

        lags = []
        changes = []
        for n in CHANGE_FIELDS:
            if n not in smart:
                continue
            col = f"smart_{n}_raw_num"
            lags.extend(
                [
                    f"lag({col}) OVER (PARTITION BY serial_number ORDER BY obs_date) AS prev_{n}",
                    f"lag(obs_date) OVER (PARTITION BY serial_number ORDER BY obs_date) AS prev_date_{n}",
                ]
            )
            changes.append(
                f"CASE WHEN date_diff('day', prev_date_{n}, obs_date) BETWEEN 1 AND 7 "
                f"THEN {col} - prev_{n} ELSE NULL END AS smart_{n}_change"
            )

        sql = f"""
            COPY (
              WITH raw AS (
                SELECT date, serial_number, filename,
                       try_cast(date AS DATE) AS obs_date,
                       {typed}
                FROM read_parquet('{src}')
              ), lagged AS (
                SELECT *, {', '.join(lags)}
                FROM raw
              ), windowed AS (
                SELECT *, {', '.join(rolling)}, {', '.join(changes)}
                FROM lagged
              )
              SELECT w.date, w.obs_date, w.serial_number, w.filename,
                     w.model, w.capacity_bytes,
                     {', '.join(f'w.smart_{n}_raw_num, w.smart_{n}_missing' for n in smart)},
                     {', '.join(f'w.smart_{n}_{stat}_{days}d' for n in smart for days in WINDOW_DAYS for stat in ('mean','max','std','count'))},
                     {', '.join(f'w.smart_{n}_change' for n in CHANGE_FIELDS if n in smart)},
                     l.failure_date, l.will_fail_30d, l.label_status
              FROM windowed w
              INNER JOIN read_parquet('{lab}') l
                ON w.date = l.date AND w.serial_number = l.serial_number
              WHERE l.will_fail_30d IS NOT NULL
            ) TO '{dest}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
        con.execute(sql)
        count = con.execute(f"SELECT count(*) FROM read_parquet('{dest}')").fetchone()[0]
        print(f"Wrote {count:,} eligible feature rows from {len(smart)} SMART attributes to {output}")
    finally:
        con.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/processed/drive_days.parquet"))
    parser.add_argument("--labels", type=Path, default=Path("data/processed/labeled.parquet"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/features.parquet"))
    args = parser.parse_args()
    build_features(args.input, args.labels, args.output)


if __name__ == "__main__":
    main()

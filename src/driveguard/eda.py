"""Generate a lightweight, reproducible data profile and EDA charts."""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import pandas as pd


def make_report(source: Path, features: Path, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    try:
        src, feat = str(source).replace("'", "''"), str(features).replace("'", "''")
        totals = con.execute(
            f"SELECT count(*) AS n_rows, count(DISTINCT serial_number) AS n_drives, "
            f"min(try_cast(date AS DATE)) AS first_day, max(try_cast(date AS DATE)) AS last_day "
            f"FROM read_parquet('{src}')"
        ).fetchone()
        labels = con.execute(
            f"SELECT count(*) AS n_rows, sum(will_fail_30d) AS positives, "
            f"sum(CASE WHEN will_fail_30d=0 THEN 1 ELSE 0 END) negatives, "
            f"round(avg(will_fail_30d)*100, 4) prevalence_pct "
            f"FROM read_parquet('{feat}')"
        ).fetchone()
        daily = con.execute(
            f"SELECT try_cast(date AS DATE) AS date, count(*) AS row_count "
            f"FROM read_parquet('{src}') GROUP BY 1 ORDER BY 1"
        ).df()
        monthly = con.execute(
            f"SELECT year(obs_date) AS year, month(obs_date) AS month, count(*) AS row_count, "
            f"sum(will_fail_30d) AS positives, avg(will_fail_30d)*100 AS prevalence_pct "
            f"FROM read_parquet('{feat}') GROUP BY 1,2 ORDER BY 1,2"
        ).df()
        models = con.execute(
            f"SELECT model, count(*) AS row_count, sum(will_fail_30d) AS positives "
            f"FROM read_parquet('{feat}') GROUP BY 1 ORDER BY row_count DESC LIMIT 10"
        ).df()
        missing = con.execute(
            f"SELECT " + ", ".join(
                f"avg(CASE WHEN try_cast(smart_{n}_raw AS DOUBLE) IS NULL THEN 1.0 ELSE 0.0 END)*100 "
                f"AS smart_{n}_missing_pct"
                for n in (5, 9, 187, 194, 197, 198)
            ) + f" FROM read_parquet('{src}')"
        ).fetchone()
    finally:
        con.close()

    low_population = daily.loc[daily["row_count"] < 5000]
    low_start = pd.to_datetime(low_population["date"]).min().date() if not low_population.empty else None
    low_end = pd.to_datetime(low_population["date"]).max().date() if not low_population.empty else None

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    axes[0].plot(pd.to_datetime(daily["date"]), daily["row_count"], color="#245b78", linewidth=1)
    axes[0].set(title="Daily source population", ylabel="Drive-day rows")
    axes[0].grid(alpha=0.25)
    months = [f"{int(r.year)}-{int(r.month):02d}" for r in monthly.itertuples()]
    axes[1].bar(months, monthly["prevalence_pct"], color="#d17a36")
    axes[1].set(title="Eligible drive-day failure prevalence", ylabel="Positive labels (%)")
    axes[1].tick_params(axis="x", rotation=40)
    axes[1].grid(axis="y", alpha=0.25)
    fig.savefig(outdir / "data_profile.png", dpi=160)
    plt.close(fig)

    split_rows = con_split_counts(features)
    report = [
        "# DriveGuard data profile",
        "",
        "## Coverage",
        "",
        f"- Source drive-day rows: {totals[0]:,}",
        f"- Distinct serial numbers: {totals[1]:,}",
        f"- Observed date range: {totals[2]} to {totals[3]}",
        f"- Eligible labeled rows: {labels[0]:,}",
        f"- Positive 30-day labels: {labels[1]:,} ({labels[3]:.4f}%)",
        f"- Negative labels: {labels[2]:,}",
        "",
            "## Monthly label counts",
            "",
            "December prevalence is based only on rows with observable 30-day outcomes near the dataset boundary; it is not directly comparable with earlier months.",
            "",
            "| Month | Eligible rows | Positives | Positive prevalence |",
        "|---|---:|---:|---:|",
    ]
    for r in monthly.itertuples():
        report.append(f"| {int(r.year)}-{int(r.month):02d} | {r.row_count:,} | {r.positives:,} | {r.prevalence_pct:.3f}% |")
    report.extend(
        [
            "",
            "## Temporal split summary",
            "",
            "The split uses a 30-day label embargo and keeps the low-population interval outside the main evaluation windows.",
            "",
            "| Split | Eligible rows | Positives | Prevalence |",
            "|---|---:|---:|---:|",
        ]
    )
    report.extend(f"| {name} | {rows:,} | {positives:,} | {prevalence:.3f}% |" for name, rows, positives, prevalence in split_rows)
    report.extend(["", "## SMART missingness", "", "| Attribute | Missing (%) |", "|---|---:|"])
    for n, pct in zip((5, 9, 187, 194, 197, 198), missing):
        report.append(f"| SMART {n} raw | {pct:.2f}% |")
    report.extend(["", "## Frequent drive models", "", "| Model | Eligible rows | Positive labels |", "|---|---:|---:|"])
    report.extend(f"| {r.model} | {r.row_count:,} | {r.positives:,} |" for r in models.itertuples())
    report.extend(
        [
            "",
            "## Data quality interpretation",
            "",
            f"Daily source population is below 5,000 rows from {low_start} through {low_end}. Counts are roughly 720–900 rows per day in this interval, compared with thousands to tens of thousands on most surrounding dates. This report records the discontinuity but does not infer its cause. The main temporal evaluation leaves this interval out; model results still describe only the retained date ranges.",
            "SMART 187 and SMART 198 raw values are fully missing in this archive and are excluded from the feature table. Other listed SMART values have usable coverage; their missingness indicators remain available to the model.",
            "",
            "See `data_profile.png` for daily row counts and monthly target prevalence.",
        ]
    )
    (outdir / "data_profile.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Wrote EDA report and chart to {outdir}")


def con_split_counts(features: Path):
    con = duckdb.connect()
    try:
        p = str(features).replace("'", "''")
        return con.execute(
            f"""SELECT split, count(*) AS n, sum(will_fail_30d) AS positives,
                       avg(will_fail_30d)*100 AS prevalence
                FROM (
                  SELECT *, CASE
                    WHEN obs_date < DATE '2013-07-01' THEN 'train'
                    WHEN obs_date >= DATE '2013-07-31' AND obs_date < DATE '2013-08-18' THEN 'validation'
                    WHEN obs_date >= DATE '2013-10-15' AND obs_date <= DATE '2013-12-01' THEN 'test'
                    ELSE 'purged_or_outside' END AS split
                  FROM read_parquet('{p}')
                ) GROUP BY split ORDER BY split"""
        ).fetchall()
    finally:
        con.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/processed/drive_days.parquet"))
    parser.add_argument("--features", type=Path, default=Path("data/processed/features.parquet"))
    parser.add_argument("--outdir", type=Path, default=Path("reports/eda"))
    args = parser.parse_args()
    make_report(args.input, args.features, args.outdir)


if __name__ == "__main__":
    main()

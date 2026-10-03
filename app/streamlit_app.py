"""A polished, explainable retrospective DriveGuard analytics experience."""

from __future__ import annotations

import json
import io
import re
from html import unescape
from html.parser import HTMLParser
from collections import OrderedDict
from pathlib import Path
import zipfile
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from datetime import date


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
METRICS_PATH = REPORTS / "model_evaluation" / "metrics.json"
PREDICTIONS_PATH = REPORTS / "model_evaluation" / "test_predictions.parquet"
SAMPLE_PATH = REPORTS / "model_evaluation" / "test_predictions_sample.parquet"
ANOMALY_PATH = REPORTS / "model_evaluation" / "isolation_forest_metrics.json"
EXPLAIN_PATH = REPORTS / "explainability"
BACKBLAZE_STATS_URL = "https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data"

st.set_page_config(
    page_title="DriveGuard | Fleet intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

PALETTE = {
    "ink": "#e8f1f7",
    "muted": "#90a4b7",
    "surface": "#101e2b",
    "surface2": "#142535",
    "line": "rgba(151, 183, 205, .14)",
    "cyan": "#55d6d0",
    "blue": "#75a9ff",
    "amber": "#ffc36b",
    "red": "#ff7b86",
    "green": "#7de0ad",
}

st.markdown(
    """
<style>
:root { --dg-bg:#09131d; --dg-card:#101e2b; --dg-card2:#142535; --dg-line:rgba(151,183,205,.14); --dg-text:#e8f1f7; --dg-muted:#90a4b7; --dg-cyan:#55d6d0; }
.stApp { background:radial-gradient(ellipse at 82% 0%, rgba(30,86,105,.22), transparent 34%), var(--dg-bg); color:var(--dg-text); font-family:'Segoe UI',system-ui,sans-serif; }
.block-container { max-width:1480px; padding-top:1.65rem; padding-bottom:3rem; }
[data-testid="stHeader"] { background:transparent; }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#0e1b28,#0a141e); border-right:1px solid var(--dg-line); }
[data-testid="stSidebar"] * { font-family:'Segoe UI',system-ui,sans-serif; }
h1,h2,h3 { color:var(--dg-text); letter-spacing:-.035em; }
h1 { font-weight:800; }
p,li,label,.stCaption { color:#bdcbd6; }
div[data-testid="stMetric"] { background:linear-gradient(145deg,rgba(20,37,53,.98),rgba(15,29,42,.94)); border:1px solid var(--dg-line); border-radius:16px; padding:17px 19px; box-shadow:0 10px 30px rgba(0,0,0,.12); min-height:118px; }
[data-testid="stMetricLabel"] { color:#9bb1c3!important; font-size:.78rem!important; text-transform:uppercase; letter-spacing:.09em; }
[data-testid="stMetricValue"] { color:#f0f7fc!important; font-weight:800; }
[data-testid="stMetricDelta"] { font-family:'DM Mono',monospace; }
.hero { position:relative; overflow:hidden; border:1px solid rgba(85,214,208,.22); border-radius:24px; padding:28px 32px; margin:0 0 23px 0; background:linear-gradient(115deg,rgba(19,47,60,.98),rgba(16,30,43,.94) 59%,rgba(22,45,60,.9)); box-shadow:0 22px 55px rgba(0,0,0,.18); }
.hero:after { content:''; position:absolute; width:300px;height:300px;border:1px solid rgba(85,214,208,.11);border-radius:50%;right:-80px;top:-160px;box-shadow:0 0 0 28px rgba(85,214,208,.025),0 0 0 56px rgba(85,214,208,.02); }
.hero-kicker { color:var(--dg-cyan); font:500 11px 'DM Mono',monospace; text-transform:uppercase; letter-spacing:.18em; margin-bottom:10px; }
.hero-title { color:#f0f8fc; font-size:clamp(27px,3.5vw,42px); line-height:1.12; font-weight:800; letter-spacing:-.05em; margin:0; }
.hero-sub { color:#a9bdca; font-size:14px; margin:10px 0 0 0; max-width:730px; line-height:1.65; }
.hero-pill { display:inline-flex; align-items:center; gap:8px; color:#ffda9b; background:rgba(255,195,107,.09); border:1px solid rgba(255,195,107,.24); padding:8px 12px; border-radius:999px; font:500 11px 'DM Mono',monospace; margin-top:17px; }
.pulse { width:7px;height:7px;background:#ffc36b;border-radius:50%;box-shadow:0 0 0 0 rgba(255,195,107,.6);animation:pulse 2s infinite; }
@keyframes pulse { 0%{box-shadow:0 0 0 0 rgba(255,195,107,.55)} 70%{box-shadow:0 0 0 8px rgba(255,195,107,0)} 100%{box-shadow:0 0 0 0 rgba(255,195,107,0)} }
.section-head { display:flex; align-items:center; gap:11px; margin:22px 0 12px; }
.section-icon { width:33px;height:33px;display:inline-flex;align-items:center;justify-content:center;border-radius:10px;color:var(--dg-cyan);background:rgba(85,214,208,.1);border:1px solid rgba(85,214,208,.16);font-size:16px; }
.section-title { font-weight:750;font-size:18px;color:#eaf3f8;letter-spacing:-.03em; }
.section-sub { color:#8fa6b7;font-size:12px;margin-left:auto; }
.insight { background:linear-gradient(110deg,rgba(85,214,208,.075),rgba(117,169,255,.045)); border-left:3px solid var(--dg-cyan); border-radius:0 12px 12px 0; padding:13px 16px; color:#c4d5df; font-size:13px; line-height:1.6; margin:9px 0 17px; }
.insight strong { color:#f0f8fc; }
.mini-card { background:var(--dg-card); border:1px solid var(--dg-line); border-radius:15px; padding:17px 18px; height:100%; }
.mini-label { color:var(--dg-muted); text-transform:uppercase; letter-spacing:.11em; font:500 10px 'DM Mono',monospace; }
.mini-value { color:#f0f7fc; font-size:21px; font-weight:800; margin-top:8px; }
.mini-note { color:#92a8b9; font-size:11px; margin-top:4px; line-height:1.45; }
.mono { font-family:'DM Mono',monospace; }
.stTabs [data-baseweb="tab-list"] { gap:7px; background:#0d1924; padding:7px; border:1px solid var(--dg-line); border-radius:14px; }
.stTabs [data-baseweb="tab"] { color:#9cb0c0; border-radius:9px; padding:9px 15px; height:auto; }
.stTabs [aria-selected="true"] { background:#1a3444!important; color:#eaf7fa!important; }
.stTabs [data-baseweb="tab-highlight"] { background:var(--dg-cyan); }
[data-testid="stDataFrame"] { border:1px solid var(--dg-line); border-radius:12px; overflow:hidden; }
div[data-testid="stExpander"] { background:rgba(16,30,43,.65); border:1px solid var(--dg-line); border-radius:12px; }
.stAlert { border-radius:12px; }
.stButton button { border-radius:10px; font-weight:700; }
.stSelectbox div[data-baseweb="select"] > div, .stTextInput input { background:#122331; border-color:var(--dg-line); }
hr { border-color:var(--dg-line); }
@media (prefers-reduced-motion: reduce) { *,*::before,*::after { animation-duration:.01ms!important; animation-iteration-count:1!important; scroll-behavior:auto!important; } }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Loading the historical evaluation…")
def load_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@st.cache_data(show_spinner="Loading complete held-out predictions…")
def load_predictions(path: str) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame["obs_date"] = pd.to_datetime(frame["obs_date"])
    frame["failure_date"] = pd.to_datetime(frame["failure_date"], errors="coerce")
    return frame


@st.cache_data(show_spinner="Loading the historical drive sample…")
def load_sample(path: str) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame["obs_date"] = pd.to_datetime(frame["obs_date"])
    frame["failure_date"] = pd.to_datetime(frame["failure_date"], errors="coerce")
    return frame


class _PageText(HTMLParser):
    """Collect readable page text without adding a scraping dependency."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        text = data.strip()
        if text:
            self.parts.append(text)


class _HttpRangeReader(io.RawIOBase):
    """Seekable view of a public ZIP archive, fetching only requested byte ranges."""

    def __init__(self, url: str, *, block_size: int = 1 << 20, cached_blocks: int = 8) -> None:
        super().__init__()
        self.url = url
        self.position = 0
        self.block_size = block_size
        self.cached_blocks = cached_blocks
        self.blocks: OrderedDict[int, bytes] = OrderedDict()
        request = Request(url, method="HEAD", headers={"User-Agent": "DriveGuard-portfolio-dashboard/1.0"})
        with urlopen(request, timeout=30) as response:
            self.size = int(response.headers["Content-Length"])
            if response.headers.get("Accept-Ranges", "").lower() != "bytes":
                raise OSError("The Backblaze archive does not support byte-range reads.")

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        target = offset if whence == io.SEEK_SET else self.position + offset if whence == io.SEEK_CUR else self.size + offset
        if target < 0:
            raise ValueError("Cannot seek before the start of the Backblaze archive.")
        self.position = target
        return self.position

    def _read_block(self, block_number: int) -> bytes:
        if block_number in self.blocks:
            self.blocks.move_to_end(block_number)
            return self.blocks[block_number]
        start = block_number * self.block_size
        end = min(self.size, start + self.block_size) - 1
        request = Request(
            self.url,
            headers={
                "Range": f"bytes={start}-{end}",
                "User-Agent": "DriveGuard-portfolio-dashboard/1.0",
            },
        )
        with urlopen(request, timeout=30) as response:
            if response.status != 206:
                raise OSError("Backblaze did not honor a byte-range request.")
            data = response.read()
        if len(data) != end - start + 1:
            raise OSError("Backblaze returned an incomplete archive range.")
        self.blocks[block_number] = data
        while len(self.blocks) > self.cached_blocks:
            self.blocks.popitem(last=False)
        return data

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            raise OSError("Unbounded archive reads are disabled.")
        remaining = min(size, max(0, self.size - self.position))
        chunks: list[bytes] = []
        while remaining:
            block_number = self.position // self.block_size
            block_offset = self.position % self.block_size
            block = self._read_block(block_number)
            count = min(remaining, len(block) - block_offset)
            chunks.append(block[block_offset : block_offset + count])
            self.position += count
            remaining -= count
        return b"".join(chunks)

    def readinto(self, buffer: bytearray) -> int:
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)


@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def load_latest_backblaze_snapshot() -> dict:
    """Read the latest aggregate published snapshot from Backblaze's data page."""
    request = Request(
        BACKBLAZE_STATS_URL,
        headers={"User-Agent": "DriveGuard-portfolio-dashboard/1.0"},
    )
    with urlopen(request, timeout=12) as response:
        html = response.read(8_000_000).decode("utf-8", errors="replace")
    parser = _PageText()
    parser.feed(html)
    text = re.sub(r"\s+", " ", unescape(" ".join(parser.parts)))

    period = re.search(r"Drive Stats (Q[1-4] 20\d{2}) Snapshot", text)
    if not period:
        raise ValueError("The Backblaze page format changed; the latest quarter could not be identified.")
    quarter = period.group(1)

    def metric(pattern: str, label: str) -> int:
        match = re.search(pattern, text, re.IGNORECASE)
        if not match:
            raise ValueError(f"The Backblaze snapshot did not include {label}.")
        return int(match.group(1).replace(",", ""))

    drive_count = metric(r"Drive count\s+([\d,]+)", "drive count")
    drive_failures = metric(r"Drive failures\s+([\d,]+)", "drive failures")
    drive_days = metric(r"Drive days\s+([\d,]+)", "drive days")
    row = re.search(
        rf"{re.escape(quarter)}\s+(?:{re.escape(quarter)}\s+)?([\d,]+)\s+(?:[\d,]+\s+)?([\d,]+)\s+(?:[\d,]+\s+)?([\d.]+%)",
        text,
    )
    if not row:
        raise ValueError(f"The Backblaze page did not include the {quarter} reliability row.")
    return {
        "quarter": quarter,
        "drive_count": drive_count,
        "drive_failures": drive_failures,
        "drive_days": drive_days,
        "quarter_afr": row.group(3),
    }


@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def load_latest_backblaze_records(quarter: str) -> tuple[pd.DataFrame, date]:
    """Read the final CSV entry of Backblaze's quarterly ZIP using HTTP range requests."""
    match = re.fullmatch(r"Q([1-4]) (20\d{2})", quarter)
    if not match:
        raise ValueError(f"Unexpected Backblaze quarter: {quarter}")
    quarter_number, year = int(match.group(1)), int(match.group(2))
    archive_url = f"https://f001.backblazeb2.com/file/Backblaze-Hard-Drive-Data/data_Q{quarter_number}_{year}.zip"
    prefix = f"data_Q{quarter_number}_{year}/"
    columns = [
        "date", "serial_number", "model", "capacity_bytes", "failure",
        "smart_5_raw", "smart_9_raw", "smart_187_raw", "smart_188_raw",
        "smart_194_raw", "smart_197_raw", "smart_198_raw",
    ]

    with zipfile.ZipFile(_HttpRangeReader(archive_url)) as archive:
        members = sorted(
            name for name in archive.namelist()
            if name.startswith(prefix) and re.search(r"/20\d{2}-\d{2}-\d{2}\.csv$", name)
        )
        if not members:
            raise ValueError(f"No daily CSV files were found in the {quarter} archive.")
        member = members[-1]
        latest_date = date.fromisoformat(Path(member).stem)
        with archive.open(member) as csv_file:
            frame = pd.read_csv(csv_file, usecols=columns)
    frame["date"] = pd.to_datetime(frame["date"])
    frame["failure"] = frame["failure"].fillna(0).astype(bool)
    return frame, latest_date


def icon_header(icon: str, title: str, subtitle: str = "") -> None:
    st.markdown(
        f'<div class="section-head"><span class="section-icon">{icon}</span>'
        f'<span class="section-title">{title}</span><span class="section-sub">{subtitle}</span></div>',
        unsafe_allow_html=True,
    )


def insight(text: str) -> None:
    st.markdown(f'<div class="insight">{text}</div>', unsafe_allow_html=True)


def style_plot(fig: go.Figure, height: int = 350) -> go.Figure:
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Manrope, sans-serif", "color": PALETTE["muted"], "size": 11},
        margin={"l": 12, "r": 12, "t": 42, "b": 12},
        legend={"orientation": "h", "y": 1.12, "x": 0, "font": {"size": 10}},
        hoverlabel={"bgcolor": "#142535", "bordercolor": "#315064", "font": {"color": "#e8f1f7"}},
    )
    fig.update_xaxes(gridcolor="rgba(151,183,205,.08)", zerolinecolor="rgba(151,183,205,.1)")
    fig.update_yaxes(gridcolor="rgba(151,183,205,.08)", zerolinecolor="rgba(151,183,205,.1)")
    return fig


def full_test_metrics(metrics: dict, anomaly: dict | None) -> pd.DataFrame:
    rows = []
    for name, result in metrics["models"].items():
        m = result["test"]
        rows.append({
            "Model": name.replace("_", " ").title(),
            "Average precision": m["average_precision"],
            "Recall": m["recall"],
            "Precision": m["precision"],
            "False-positive rate": m["false_positive_rate"],
            "Alerts / 1,000": m["alerts_per_1000_drive_days"],
            "Failure-event coverage": m["failure_events_with_any_alert_recall"],
            "Median lead (days)": m["median_lead_days_detected_events"],
        })
    if anomaly:
        m = anomaly["test"]
        rows.append({
            "Model": "Isolation Forest · anomaly rank",
            "Average precision": m["average_precision"], "Recall": m["recall"],
            "Precision": m["precision"], "False-positive rate": m["false_positive_rate"],
            "Alerts / 1,000": m["alerts_per_1000_drive_days"],
            "Failure-event coverage": m["failure_events_with_any_alert_recall"],
            "Median lead (days)": m["median_lead_days_detected_events"],
        })
    return pd.DataFrame(rows)


missing = [p for p in (METRICS_PATH, PREDICTIONS_PATH, SAMPLE_PATH) if not p.exists()]
if missing:
    st.error("Dashboard data is incomplete. Run the data, training, evaluation, and score-sample commands first.")
    for path in missing:
        st.code(str(path.relative_to(ROOT)))
    st.stop()

metrics = load_json(str(METRICS_PATH))
predictions = load_predictions(str(PREDICTIONS_PATH))
sample = load_sample(str(SAMPLE_PATH))
model_metrics = metrics["models"]
xgb_result = model_metrics["xgboost"]
xgb_test = xgb_result["test"]
validation_threshold = float(xgb_test["threshold_selected_on_validation_at_max_1pct_fpr"])
anomaly = load_json(str(ANOMALY_PATH)) if ANOMALY_PATH.exists() else None

# Sidebar: clear scope, filters, and a short legend.
with st.sidebar:
    st.markdown("<div class='hero-kicker'>DRIVEGUARD / HISTORICAL LAB</div>", unsafe_allow_html=True)
    st.markdown("### Explore the dashboard")
    dates = predictions["obs_date"].dt.date
    min_date, max_date = min(dates), max(dates)
    selected_dates = st.date_input("Observation dates", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
        start_date, end_date = selected_dates
    else:
        start_date, end_date = min_date, max_date
    model_options = ["All drive models"] + sorted(sample["model"].dropna().astype(str).unique().tolist())
    selected_model = st.selectbox("Drive model", model_options)
    st.markdown("---")
    st.markdown("**How to read risk**")
    st.caption("The XGBoost score ranks 2013 historical drive-days. It is not a live reading or a promise of failure.")
    st.markdown("<span class='mono' style='color:#ffc36b'>──</span> Validation-selected reference threshold", unsafe_allow_html=True)
    st.caption(f"Reference threshold: {validation_threshold:.3f} · validation FPR target ≤ 1%")
    st.markdown("---")
    st.caption("Model: Backblaze 2013 · held-out dates: 2013-10-15 to 2013-12-01 · horizon: 30 days")
    st.caption("Latest public fleet snapshot is shown separately and updates when Backblaze publishes it.")
    st.caption("This app is a research demo, not a live drive monitor or production alerting service.")

st.markdown(
    "<div class='hero'><div class='hero-kicker'>PREDICTIVE MAINTENANCE · RESEARCH EDITION</div>"
    "<div class='hero-title'>DriveGuard <span style='color:#55d6d0'>Fleet intelligence</span></div>"
    "<div class='hero-sub'>Explore how historical SMART telemetry ranked drive-days ahead of recorded failures."
    " The historical model and the latest public fleet snapshot are separated so their dates and limits stay clear.</div>"
    "<div class='hero-pill'><span class='pulse'></span> HISTORICAL MODEL · PUBLIC FLEET SNAPSHOT</div></div>",
    unsafe_allow_html=True,
)

tab_overview, tab_live, tab_risk, tab_drive, tab_models, tab_explain, tab_quality = st.tabs([
    "✦  Fleet overview", "↻  Latest fleet data", "⌁  Risk explorer", "◉  Drive detail", "▥  Model lab", "✧  Why this score", "◎  Data quality",
])

with tab_overview:
    n = int(xgb_test["rows"])
    positives = int(xgb_test["positive_rows"])
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Held-out drive-days", f"{n:,}", help="Complete test set; no healthy-row sampling.")
    col2.metric("Recorded failures in horizon", f"{positives:,}", help="Positive rows label a failure 1–30 days after the observation.")
    col3.metric("Observed prevalence", f"{xgb_test['prevalence']:.3%}", help="Natural test-set prevalence; not the artificially enriched drill-down sample.")
    col4.metric("XGBoost average precision", f"{xgb_test['average_precision']:.3f}", help="Precision-recall summary; compare to prevalence and the other methods.")
    col5.metric("Failure-event coverage", f"{xgb_test['failure_events_with_any_alert_recall']:.1%}", help="Share of eligible failure events with at least one thresholded alert row.")

    icon_header("⌁", "What happened across the test window", "Daily volume and labelled events")
    daily = predictions.groupby("obs_date", as_index=False).agg(
        drive_days=("will_fail_30d", "size"), positive_rows=("will_fail_30d", "sum")
    )
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily["obs_date"], y=daily["drive_days"], name="Observed drive-days", mode="lines", line={"color": PALETTE["blue"], "width": 2}, fill="tozeroy", fillcolor="rgba(117,169,255,.08)"))
    fig.add_trace(go.Scatter(x=daily["obs_date"], y=daily["positive_rows"], name="Positive drive-day labels", mode="lines", line={"color": PALETTE["amber"], "width": 2}, yaxis="y2"))
    fig.update_layout(
        title="Daily test-set coverage",
        yaxis={"title": {"text": "Drive-days", "font": {"color": PALETTE["blue"]}}},
        yaxis2={"title": {"text": "Positive rows", "font": {"color": PALETTE["amber"]}}, "overlaying": "y", "side": "right"},
    )
    st.plotly_chart(style_plot(fig, 330), use_container_width=True)
    insight("<strong>How to interpret this:</strong> each point represents a drive observed on a calendar day. Positive rows overlap within failure events, so positive drive-days are not a count of unique failed drives.")

    icon_header("◌", "Model in one minute", "Key operating point")
    a, b, c = st.columns([1.1, 1, 1])
    a.markdown(f"<div class='mini-card'><div class='mini-label'>Validation-selected threshold</div><div class='mini-value mono'>{validation_threshold:.4f}</div><div class='mini-note'>Selected to target ≤1% false-positive rate on validation data.</div></div>", unsafe_allow_html=True)
    b.markdown(f"<div class='mini-card'><div class='mini-label'>Test alerts per 1,000 rows</div><div class='mini-value'>{xgb_test['alerts_per_1000_drive_days']:.1f}</div><div class='mini-note'>At this historical threshold, measured on the complete test set.</div></div>", unsafe_allow_html=True)
    c.markdown(f"<div class='mini-card'><div class='mini-label'>Median detected-event lead</div><div class='mini-value'>{xgb_test['median_lead_days_detected_events']:.0f} days</div><div class='mini-note'>Among detected eligible events; missed events are not included.</div></div>", unsafe_allow_html=True)

    with st.expander("What does average precision mean?"):
        st.write("Average precision summarizes precision across recall levels. It is useful for rare events because it focuses on how well positive cases are ranked. The test prevalence is the no-skill reference (about 0.005 here); average precision is not the same as accuracy or a calibrated probability.")

with tab_live:
    icon_header("↻", "Latest drive-level snapshot", "Latest published quarter · actual public SMART rows")
    st.markdown(
        "This view loads individual daily SMART records from Backblaze’s latest published quarter. "
        "Backblaze publishes the data in quarterly releases, so it is **not real time "
        "and does not represent drives connected to your own computer**."
    )
    try:
        snapshot = load_latest_backblaze_snapshot()
        with st.spinner(f"Querying the {snapshot['quarter']} public drive records…"):
            current_drives, observed_date = load_latest_backblaze_records(snapshot["quarter"])
        st.caption(f"Published quarter: **{snapshot['quarter']}** · latest daily snapshot: **{observed_date:%B %d, %Y}** · cached for up to 6 hours")
        live_a, live_b, live_c, live_d = st.columns(4)
        live_a.metric("Drive records that day", f"{len(current_drives):,}")
        live_b.metric("Failures recorded that day", f"{int(current_drives['failure'].sum()):,}")
        live_c.metric("Drive models", f"{current_drives['model'].nunique():,}")
        live_d.metric("Quarterly AFR", snapshot["quarter_afr"], help="Backblaze's aggregate annualized failure rate for the published quarter; not an individual-drive probability.")

        current_drives["capacity_tb"] = current_drives["capacity_bytes"] / 1_000_000_000_000
        top_models = (
            current_drives.groupby("model", dropna=False)
            .agg(drive_records=("serial_number", "size"), failures=("failure", "sum"))
            .nlargest(12, "drive_records")
            .sort_values("drive_records")
            .reset_index()
        )
        left, right = st.columns([1.15, 1])
        with left:
            model_fig = px.bar(
                top_models, x="drive_records", y="model", orientation="h", color="failures",
                color_continuous_scale=["#55d6d0", "#ff7b86"],
                title=f"Most represented drive models · {observed_date:%Y-%m-%d}",
                labels={"drive_records": "Drive records", "model": "Drive model", "failures": "Failures"},
            )
            st.plotly_chart(style_plot(model_fig, 390), use_container_width=True)
        with right:
            st.markdown("#### Browse the actual 2026 rows")
            model_choices = ["All models"] + sorted(current_drives["model"].dropna().astype(str).unique())
            chosen_live_model = st.selectbox("Drive model", model_choices, key="live_model")
            failure_choice = st.selectbox("Snapshot status", ["All records", "Failure recorded", "No failure recorded"], key="live_failure")
            serial_search = st.text_input("Find a serial number", key="live_serial").strip()
            live_rows = current_drives.copy()
            if chosen_live_model != "All models":
                live_rows = live_rows[live_rows["model"].astype(str).eq(chosen_live_model)]
            if failure_choice == "Failure recorded":
                live_rows = live_rows[live_rows["failure"].eq(1)]
            elif failure_choice == "No failure recorded":
                live_rows = live_rows[live_rows["failure"].eq(0)]
            if serial_search:
                live_rows = live_rows[live_rows["serial_number"].astype(str).str.contains(serial_search, case=False, na=False, regex=False)]
            st.caption(f"{len(live_rows):,} matching records · these are the complete daily rows for the displayed snapshot.")

        display_columns = {
            "date": "Snapshot date", "serial_number": "Serial number", "model": "Drive model",
            "capacity_tb": "Capacity (TB)", "failure": "Failure recorded",
            "smart_5_raw": "Reallocated sectors · SMART 5", "smart_9_raw": "Power-on hours · SMART 9",
            "smart_187_raw": "Reported uncorrectable · SMART 187", "smart_188_raw": "Command timeout · SMART 188",
            "smart_194_raw": "Temperature · SMART 194", "smart_197_raw": "Pending sectors · SMART 197",
            "smart_198_raw": "Uncorrectable sectors · SMART 198",
        }
        shown = live_rows.sort_values(["failure", "smart_5_raw"], ascending=[False, False], na_position="last")
        st.dataframe(
            shown[list(display_columns)].rename(columns=display_columns).head(500),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Snapshot date": st.column_config.DateColumn(format="YYYY-MM-DD"),
                "Capacity (TB)": st.column_config.NumberColumn(format="%.2f TB"),
                "Failure recorded": st.column_config.CheckboxColumn(),
            },
        )
        st.download_button(
            "Download first 10,000 matching records as CSV",
            data=shown[list(display_columns)].rename(columns=display_columns).head(10_000).to_csv(index=False).encode("utf-8"),
            file_name=f"driveguard_backblaze_{observed_date:%Y%m%d}.csv",
            mime="text/csv",
            use_container_width=True,
        )
        insight(
            "<strong>Model boundary:</strong> these 2026 rows are not scored by the 2013 XGBoost model. "
            "The SMART fields are raw vendor-reported values and can mean different things across drive models. "
            "A failure flag records an outcome on that day; it is not a prediction."
        )
    except Exception as exc:
        st.error("The latest per-drive snapshot could not be loaded. The historical model dashboard remains available.")
        st.caption(f"Backblaze source request failed ({type(exc).__name__}). Try again after the source is reachable.")
    st.markdown(f"[Open Backblaze’s official Drive Stats page ↗]({BACKBLAZE_STATS_URL})")
    st.caption("The dashboard reads the latest daily CSV from Backblaze's public quarterly archive using HTTP byte ranges; it fetches only that day’s compressed file, not the full archive. Results are cached for six hours.")

with tab_risk:
    icon_header("⌁", "Historical risk explorer", "Complete-test operating view")
    filtered = predictions[predictions["obs_date"].dt.date.between(start_date, end_date)].copy()
    threshold_min = float(np.nanmin(filtered["risk_xgboost"]))
    threshold_max = float(np.nanmax(filtered["risk_xgboost"]))
    t1, t2 = st.columns([2, 1])
    with t1:
        chosen_threshold = st.slider("Inspect a score cutoff", min_value=0.0, max_value=min(1.0, max(threshold_max, .05)), value=min(max(validation_threshold, 0.0), min(1.0, max(threshold_max, .05))), step=.001, help="This recalculates descriptive metrics on the historical test set. Do not tune a deployment threshold on the test set.")
    with t2:
        st.metric("Validation reference", f"{validation_threshold:.4f}", help="Reference chosen on validation, before the held-out test period.")
    y = filtered["will_fail_30d"].astype(int).to_numpy()
    alert = filtered["risk_xgboost"].to_numpy() >= chosen_threshold
    tp = int(np.sum(alert & (y == 1))); fp = int(np.sum(alert & (y == 0)))
    fn = int(np.sum(~alert & (y == 1))); tn = int(np.sum(~alert & (y == 0)))
    fpr = fp / max(1, fp + tn); precision = tp / max(1, tp + fp); recall = tp / max(1, tp + fn)
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Historical precision", f"{precision:.1%}")
    q2.metric("Historical recall", f"{recall:.1%}")
    q3.metric("Historical false-positive rate", f"{fpr:.2%}")
    q4.metric("Flagged drive-days", f"{tp+fp:,}")
    insight("These metrics are recomputed on the selected portion of the <strong>test</strong> interval for exploration. The validation reference was chosen before test evaluation; repeatedly adjusting this control to optimize test results would turn the holdout into tuning data.")

    if threshold_min <= threshold_max:
        hist = go.Figure()
        hist.add_trace(go.Histogram(x=filtered.loc[filtered["will_fail_30d"].eq(0), "risk_xgboost"], name="No positive label", marker_color=PALETTE["blue"], opacity=.72, nbinsx=55))
        hist.add_trace(go.Histogram(x=filtered.loc[filtered["will_fail_30d"].eq(1), "risk_xgboost"], name="Positive label", marker_color=PALETTE["amber"], opacity=.78, nbinsx=55))
        hist.add_vline(x=chosen_threshold, line_color=PALETTE["red"], line_dash="dash", annotation_text="Selected cutoff", annotation_position="top")
        hist.update_layout(barmode="overlay", title="XGBoost score distribution · full test rows", xaxis_title="Model score (ranking output)", yaxis_title="Drive-days")
        st.plotly_chart(style_plot(hist, 365), use_container_width=True)

    icon_header("⌕", "Sampled drive-day records", "Drill-down sample · not a fleet denominator")
    sample_view = sample[sample["obs_date"].dt.date.between(start_date, end_date)].copy()
    if selected_model != "All drive models":
        sample_view = sample_view[sample_view["model"].astype(str).eq(selected_model)]
    sample_view = sample_view[sample_view["risk_xgboost"] >= chosen_threshold].sort_values("risk_xgboost", ascending=False)
    sample_view["Label"] = np.where(sample_view["will_fail_30d"].eq(1), "Failure recorded in next 30 days", "No positive label")
    columns = ["obs_date", "serial_number", "model", "risk_xgboost", "risk_logistic_regression", "anomaly_isolation_forest", "Label", "failure_date"]
    st.dataframe(sample_view[columns].head(500), use_container_width=True, hide_index=True, column_config={
        "obs_date": st.column_config.DateColumn("Observed on", format="YYYY-MM-DD"),
        "serial_number": st.column_config.TextColumn("Drive serial"),
        "model": st.column_config.TextColumn("Drive model"),
        "risk_xgboost": st.column_config.ProgressColumn("XGBoost score", min_value=0, max_value=1, format="%.3f"),
        "risk_logistic_regression": st.column_config.NumberColumn("Logistic score", format="%.3f"),
        "anomaly_isolation_forest": st.column_config.NumberColumn("Anomaly score", format="%.3f"),
        "failure_date": st.column_config.DateColumn("Recorded failure date", format="YYYY-MM-DD"),
    })
    st.caption("The drill-down file includes every positive-labelled test row and a deterministic sample of healthy-drive histories. Its row counts and positive share are intentionally not presented as fleet prevalence.")

with tab_drive:
    icon_header("◉", "Drive history", "Individual historical examples")
    candidate_sample = sample[sample["obs_date"].dt.date.between(start_date, end_date)].copy()
    if selected_model != "All drive models":
        candidate_sample = candidate_sample[candidate_sample["model"].astype(str).eq(selected_model)]
    serials = sorted(candidate_sample["serial_number"].dropna().astype(str).unique().tolist())
    if not serials:
        st.info("No sampled drive histories match these filters. Widen the date range or choose another model.")
    else:
        serial = st.selectbox("Choose a drive from the historical sample", serials)
        series = sample[sample["serial_number"].astype(str).eq(serial)].sort_values("obs_date")
        series = series[series["obs_date"].dt.date.between(start_date, end_date)]
        model_name = str(series["model"].dropna().iloc[0]) if series["model"].notna().any() else "Unknown model"
        is_positive = bool(series["will_fail_30d"].eq(1).any())
        state = "Recorded failure within the target horizon" if is_positive else "No positive label in the sampled interval"
        d1, d2, d3 = st.columns(3)
        d1.metric("Drive model", model_name)
        d2.metric("Observed test days", f"{len(series):,}")
        d3.metric("Label summary", "Positive history" if is_positive else "No positive label")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=series["obs_date"], y=series["risk_xgboost"], name="XGBoost score", mode="lines+markers", line={"color": PALETTE["cyan"], "width": 2}, marker={"size": 5}, hovertemplate="%{x|%Y-%m-%d}<br>Score %{y:.3f}<extra></extra>"))
        fig.add_hline(y=validation_threshold, line_color=PALETTE["amber"], line_dash="dash", annotation_text="Validation reference")
        failures = series.loc[series["failure_date"].notna(), "failure_date"]
        if not failures.empty:
            failure_date = failures.min()
            fig.add_vline(x=failure_date, line_color=PALETTE["red"], line_dash="dot", annotation_text="Recorded failure")
        fig.update_yaxes(title="Ranking score", rangemode="tozero")
        fig.update_xaxes(title="Observation date")
        st.plotly_chart(style_plot(fig, 390), use_container_width=True)
        if is_positive:
            insight(f"<strong>Observed outcome:</strong> at least one sampled drive-day for this drive was labelled positive, meaning the archive records a failure 1–30 days after that observation. This is retrospective context. {state}.")
        else:
            insight("<strong>Observed outcome:</strong> the selected sample contains no positive-labelled drive-day for this drive in the displayed period. This does not prove the drive was healthy outside the observed data or after the follow-up window.")
        with st.expander("Why can the score rise or fall from day to day?"):
            st.write("The model receives current SMART readings, missingness indicators, trailing summaries, changes, and drive-model identity. New observations can change these inputs. Raw SMART attributes are vendor/model-specific; changes in a model score are associations learned by this historical model, not causal explanations.")
        st.dataframe(series[["obs_date", "risk_xgboost", "risk_logistic_regression", "anomaly_isolation_forest", "will_fail_30d", "failure_date"]], use_container_width=True, hide_index=True)

with tab_models:
    icon_header("▥", "Model comparison lab", "Full held-out test interval")
    comparison = full_test_metrics(metrics, anomaly)
    best = comparison.sort_values("Average precision", ascending=False).iloc[0]
    st.markdown(f"<div class='insight'><strong>Top average precision:</strong> {best['Model']} at {best['Average precision']:.3f}. Read it alongside precision, alert volume, and the alert's false-positive rate; ranking quality alone does not define operating value.</div>", unsafe_allow_html=True)
    c1, c2 = st.columns([1.15, 1])
    with c1:
        fig = px.bar(comparison.sort_values("Average precision"), x="Average precision", y="Model", orientation="h", color="Model", color_discrete_sequence=[PALETTE["cyan"], PALETTE["blue"], PALETTE["amber"], PALETTE["red"]], text_auto=".3f", title="Ranking quality · average precision")
        fig.update_layout(showlegend=False, xaxis_title="Average precision", yaxis_title="")
        st.plotly_chart(style_plot(fig, 340), use_container_width=True)
    with c2:
        fig = px.scatter(comparison, x="False-positive rate", y="Recall", size="Alerts / 1,000", color="Model", hover_name="Model", hover_data={"Precision": ":.1%", "Failure-event coverage": ":.1%", "Alerts / 1,000": ":.1f"}, color_discrete_sequence=[PALETTE["cyan"], PALETTE["blue"], PALETTE["amber"], PALETTE["red"]], title="Operating trade-off · bubble size = alerts")
        fig.update_xaxes(tickformat=".1%", title="False-positive rate")
        fig.update_yaxes(tickformat=".0%", title="Row-level recall")
        st.plotly_chart(style_plot(fig, 340), use_container_width=True)
    st.dataframe(comparison.style.format({
        "Average precision": "{:.3f}", "Recall": "{:.1%}", "Precision": "{:.1%}",
        "False-positive rate": "{:.2%}", "Alerts / 1,000": "{:.1f}",
        "Failure-event coverage": "{:.1%}", "Median lead (days)": "{:.1f}",
    }), use_container_width=True, hide_index=True)
    st.caption(f"Complete test: {int(xgb_test['rows']):,} drive-days · {int(xgb_test['positive_rows']):,} positives · {xgb_test['prevalence']:.3%} prevalence. Model thresholds were selected using validation data.")
    if anomaly:
        fpr = anomaly["test"]["false_positive_rate"]
        st.warning(f"Isolation Forest is an anomaly ranking, not a calibrated failure probability. Its validation-selected threshold produced {fpr:.1%} test false-positive rate ({anomaly['test']['alerts_per_1000_drive_days']:.1f} alerts per 1,000 drive-days).")
    with st.expander("Definitions for the comparison table"):
        st.markdown("- **Average precision:** area-like summary over the precision-recall curve; compare with the rare positive prevalence.\n- **Recall:** fraction of positive drive-days above the validation-selected threshold.\n- **Precision:** share of alerted drive-days that were positive-labelled.\n- **False-positive rate:** share of negative drive-days that were alerted.\n- **Failure-event coverage:** fraction of eligible serial-number/failure-date events with at least one alert.\n- **Median lead:** median days from detected positive observations to the recorded failure among detected events.")

with tab_explain:
    icon_header("✧", "Why did the model score this way?", "Model behavior · not causation")
    global_path = EXPLAIN_PATH / "shap_global_importance.png"
    cases_path = EXPLAIN_PATH / "case_explanations.csv"
    explanation_path = EXPLAIN_PATH / "explanations.md"
    if global_path.exists():
        st.image(str(global_path), caption="Global SHAP importance on a deterministic sample from the held-out test period")
        insight("<strong>Global view:</strong> mean absolute SHAP values summarize how much features moved model scores across the sampled rows. They do not say that a feature causes failure.")
    if cases_path.exists():
        cases = pd.read_csv(cases_path)
        kind_labels = {"detected_failure": "Detected failure", "missed_failure": "Missed failure", "false_alarm": "False alarm"}
        available = cases["case_type"].astype(str).tolist()
        chosen_case = st.selectbox("Walk through a historical example", available, format_func=lambda v: kind_labels.get(v, v.replace("_", " ").title()))
        row = cases.loc[cases["case_type"].eq(chosen_case)].iloc[0]
        s1, s2, s3 = st.columns(3)
        s1.metric("Drive", str(row["serial_number"]))
        s2.metric("Observation date", str(row["observation_date"])[:10])
        s3.metric("XGBoost score", f"{float(row['xgboost_score']):.3f}")
        local_path = EXPLAIN_PATH / f"{chosen_case}.png"
        if local_path.exists():
            st.image(str(local_path), caption="Local SHAP waterfall · feature contributions relative to this model's baseline score")
        features = [str(row[f"top_feature_{i}"]) for i in range(1, 6) if pd.notna(row.get(f"top_feature_{i}"))]
        if features:
            st.markdown("**Largest score contributors in this example**")
            st.write(" · ".join(f"`{f}`" for f in features))
        st.caption("A detected failure, missed failure, or false alarm is an example selected from a deterministic dashboard sample. It is not necessarily representative of all drives.")
    if explanation_path.exists():
        with st.expander("Read the explanation notes"):
            st.markdown(explanation_path.read_text(encoding="utf-8"))
    if not global_path.exists() and not cases_path.exists():
        st.info("No explanation artifacts were found. Run the SHAP explanation phase in the README to populate this page.")
    with st.expander("What SHAP can and cannot tell us"):
        st.write("SHAP decomposes a fitted model's output relative to a background reference. Positive contributions raise this model's score and negative contributions lower it. They describe model behavior in this dataset; they do not establish physical causation, universal SMART meanings, or future-fleet reliability.")

with tab_quality:
    icon_header("◎", "Calibration, uncertainty, and data quality", "Keep model performance in context")
    analysis_path = REPORTS / "analysis" / "analysis_summary.json"
    calibration_path = REPORTS / "analysis" / "calibration_test.png"
    slices_path = REPORTS / "analysis" / "error_slices.csv"
    e1, e2 = st.columns(2)
    e1.markdown("<div class='mini-card'><div class='mini-label'>Archive coverage</div><div class='mini-value'>2013</div><div class='mini-note'>266 daily CSVs · source rows from 2013-04-10 to 2013-12-31.</div></div>", unsafe_allow_html=True)
    e2.markdown("<div class='mini-card'><div class='mini-label'>Known population discontinuity</div><div class='mini-value'>Aug 20 – Oct 14</div><div class='mini-note'>Daily row counts fall sharply; the cause is unknown. The principal validation and test windows avoid the interval.</div></div>", unsafe_allow_html=True)
    if analysis_path.exists():
        analysis = load_json(str(analysis_path))
        calibration_rows = []
        interval_rows = []
        for name, stats in analysis["models"].items():
            calibration_rows.append({"Model": name.replace("_", " ").title(), "Brier score": stats["brier_score"], "10-bin ECE": stats["expected_calibration_error_10_quantile_bins"]})
            for metric, bounds in stats["drive_cluster_bootstrap_95pct"]["intervals"].items():
                interval_rows.append({"Model": name.replace("_", " ").title(), "Measure": metric.replace("_", " ").title(), "Lower 95%": bounds["lower_95"], "Upper 95%": bounds["upper_95"]})
        icon_header("◒", "Probability calibration", "Natural-prevalence test rows")
        st.dataframe(pd.DataFrame(calibration_rows).style.format({"Brier score": "{:.5f}", "10-bin ECE": "{:.5f}"}), use_container_width=True, hide_index=True)
        if calibration_path.exists():
            st.image(str(calibration_path), caption="Observed prevalence versus mean score within quantile bins; bin sizes are similar but score widths may differ")
        st.caption("Brier score is the mean squared probability error. Expected calibration error summarizes differences between predicted scores and observed rates across bins. Neither replaces calibration review at the intended operating range.")
        icon_header("↔", "Uncertainty across drives", "Cluster bootstrap · 300 replicates")
        st.dataframe(pd.DataFrame(interval_rows).style.format({"Lower 95%": "{:.4f}", "Upper 95%": "{:.4f}"}), use_container_width=True, hide_index=True)
        st.caption(analysis.get("uncertainty_note", "Intervals quantify drive-level sampling variation within this historical test interval."))
    if slices_path.exists():
        slices = pd.read_csv(slices_path)
        icon_header("⌕", "Where errors concentrate", "Diagnostic slices")
        slice_options = sorted(slices["slice"].dropna().unique().tolist())
        if slice_options:
            slice_type = st.selectbox("Group the test results by", slice_options)
            st.dataframe(slices[slices["slice"].eq(slice_type)], use_container_width=True, hide_index=True)
            st.caption("Slice results describe variation across this test set. They were not used to retune thresholds.")
    with st.expander("Important limits before interpreting this model"):
        st.markdown("- The benchmark covers Backblaze's 2013 archive and has no current-fleet external validation.\n- SMART raw values and missingness differ across drive models.\n- Validation and test prevalence differ substantially, so a validation threshold can produce a different test alert burden.\n- Row-level alerts repeat over a drive's history and are not deduplicated into action episodes.\n- Cluster bootstrap intervals describe uncertainty across serial numbers in this test period; they do not quantify temporal or source shift.\n- The dashboard uses historical outcomes to explain past predictions and must not be used to make operational replacement decisions.")

st.markdown("<div style='text-align:center;color:#71899a;font:10px DM Mono,monospace;margin-top:34px'>DRIVEGUARD · BACKBLAZE 2013 · 30-DAY LABEL HORIZON · RETROSPECTIVE STUDY</div>", unsafe_allow_html=True)

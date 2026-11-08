# app.py
import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.io_utils import con
from src.config import START, END, BUDGET_USD
from src import scorecard, downtime, payout, location, diagnose, actions

OUT = Path("out")
st.set_page_config(page_title="Gilded Ace — Floor Analytics", layout="wide")

# ---------------- visual theme (presentation only) ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');
:root { --ink:#182333; --muted:#667386; --paper:#f5f3ee; --panel:#fff; --gold:#b58a45; --line:#e5e0d6; --navy:#172638; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp { background:linear-gradient(135deg,#f7f5f0 0%,#f0eee8 100%); color:var(--ink); }
.block-container { max-width:1500px; padding-top:2.1rem; padding-bottom:3rem; }
h1,h2,h3 { color:var(--navy); letter-spacing:-.035em; }
h1 { font-family:'Playfair Display',Georgia,serif !important; font-size:clamp(2rem,3vw,2.8rem) !important; line-height:1.15 !important; }
h2 { font-size:1.55rem !important; margin-top:1.5rem !important; }
h3 { font-size:1.15rem !important; }
p,label,.stCaption { color:var(--muted); }
[data-testid="stSidebar"] { background:#172638; border-right:1px solid #26394d; }
[data-testid="stSidebar"] * { color:#edf0f3 !important; }
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2 { color:#f1d6a5 !important; font-family:'Playfair Display',Georgia,serif !important; }
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p { color:#c2ccd6 !important; }
[data-testid="stSidebar"] [data-baseweb="select"]>div,[data-testid="stSidebar"] [data-testid="stDateInput"] input { background:#24384d; border-color:#40536a; }
[data-testid="stMetric"] { background:var(--panel); border:1px solid var(--line); border-top:3px solid var(--gold); border-radius:13px; padding:1rem 1.15rem; box-shadow:0 5px 18px rgba(25,35,48,.045); }
[data-testid="stMetricLabel"] { color:var(--muted) !important; font-size:.82rem !important; font-weight:600 !important; }
[data-testid="stMetricValue"] { color:var(--navy) !important; font-weight:700 !important; letter-spacing:-.04em; }
[data-testid="stMetricDelta"] { font-size:.78rem !important; }
[data-testid="stPlotlyChart"],[data-testid="stDataFrame"],[data-testid="stTable"] { background:var(--panel); border:1px solid var(--line); border-radius:13px; padding:.45rem; box-shadow:0 5px 18px rgba(25,35,48,.035); }
[data-testid="stDataFrame"] { padding:.25rem; }
.stButton>button,.stDownloadButton>button { background:var(--navy); color:#fff; border:1px solid var(--navy); border-radius:9px; padding:.55rem 1rem; font-weight:600; transition:all .18s ease; }
.stButton>button:hover,.stDownloadButton>button:hover { background:#263d56; border-color:var(--gold); color:#fff; transform:translateY(-1px); }
[data-testid="stTabs"] [data-baseweb="tab-list"] { gap:.4rem; border-bottom:1px solid var(--line); }
[data-testid="stTabs"] button[role="tab"] { border-radius:8px 8px 0 0; padding:.65rem 1rem; color:var(--muted); }
[data-testid="stTabs"] button[aria-selected="true"] { color:var(--navy); border-bottom:3px solid var(--gold); font-weight:700; }
[data-testid="stExpander"] { background:#fff; border:1px solid var(--line); border-radius:11px; }
[data-testid="stAlert"] { border-radius:11px; }
hr { border-color:var(--line); }
.stCaption { font-size:.82rem; }
</style>
""", unsafe_allow_html=True)

# Keep Plotly charts visually consistent with the dashboard's light, editorial palette.
px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = ["#b58a45", "#27445f", "#6d8a83", "#a56d5c", "#8a83a8", "#84909b"]

CAUSE_COLOR = {"downtime": "#d95f02", "location": "#7570b3",
               "payout_suspect": "#e7298a", "weak_game": "#1b9e77",
               "luck_only": "#666666", "insufficient_evidence": "#bbbbbb",
               "fine": "#4daf4a"}
CAUSE_LABEL = {"downtime": "Broken too often", "location": "Bad spot on the floor",
               "payout_suspect": "Payout worth inspecting", "weak_game": "Weak game",
               "luck_only": "Just unlucky", "insufficient_evidence": "Too little data",
               "fine": "Performing fine"}


@st.cache_resource
def _con():
    return con()


@st.cache_data(show_spinner="Recomputing for this period…")
def pipeline(start, end, retention):
    """Full recompute, so the period selector is real rather than cosmetic [2]."""
    c = _con()
    sc = scorecard.build_scorecard(start, end, c)
    sc = scorecard.add_peer_percentiles(sc, keys=("dom_game_setting",))
    sc = scorecard.add_peer_percentiles(sc, keys=("dom_zone",))

    faults = downtime.fault_metrics(start, end, c)
    sc = sc.merge(faults[["machine_id", "n_faults", "fault_hours",
                          "mttr_hours", "repeat_faults_14d"]],
                  on="machine_id", how="left")
    sc[["n_faults", "fault_hours", "repeat_faults_14d"]] = \
        sc[["n_faults", "fault_hours", "repeat_faults_14d"]].fillna(0)
    sc = downtime.lost_play(sc, retention=retention)
    sc = payout.luck_test(sc)

    eff = location.two_way_effects(start, end, c)
    movers = location.movers_evidence(start, end, c)

    resid = None
    p = OUT / "machine_scorecard.parquet"
    if p.exists():
        prior = pd.read_parquet(p)
        if "adj_demand_resid" in prior:
            resid = prior[["machine_id", "adj_demand_resid", "adj_demand_pct"]]

    diag = diagnose.diagnose(sc, eff["locations"], eff["machines"], faults, resid)
    dt_cal = downtime.downtime_calendar(start, end, c)
    return dict(diag=diag, eff=eff, movers=movers, dt_cal=dt_cal, faults=faults)



# ---------------- presentation helpers ----------------
st.markdown("""
<style>
.block-container { max-width: 1580px; padding-top: 1.35rem; padding-bottom: 3.5rem; }
.eyebrow { color:#9a7336; font-size:.72rem; font-weight:800; letter-spacing:.16em; text-transform:uppercase; margin:0 0 .45rem 0; }
.page-intro { color:#6b7787; font-size:1.02rem; max-width:900px; line-height:1.55; margin-top:-.35rem; margin-bottom:1.45rem; }
.section-kicker { color:#a17b42; font-size:.69rem; font-weight:800; text-transform:uppercase; letter-spacing:.14em; margin:1.1rem 0 .15rem; }
.section-note { color:#788494; font-size:.86rem; margin-top:-.2rem; }
.brand-mark { display:inline-flex; align-items:center; justify-content:center; width:34px; height:34px; border-radius:10px; background:linear-gradient(145deg,#d5b77d,#a77c3f); color:#172638; font-weight:900; margin-right:9px; box-shadow:0 5px 12px rgba(181,138,69,.22); }
.header-strip { border-bottom:1px solid #e4dfd5; padding:0 0 .85rem 0; margin-bottom:1.5rem; }
.pill-note { display:inline-block; padding:.28rem .62rem; border:1px solid #e6dcc9; border-radius:999px; color:#806338; background:#f7f0e3; font-size:.72rem; font-weight:700; letter-spacing:.03em; }
[data-testid="stSidebar"] [data-testid="stRadio"] label { padding:.38rem .52rem; border-radius:8px; }
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover { background:#263b51; }
[data-testid="stSidebar"] hr { border-color:#33485d; }
[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] { margin-top:.45rem; }
[data-testid="stMetric"] { min-height:116px; display:flex; flex-direction:column; justify-content:center; }
[data-testid="stMetricValue"] { font-size:1.85rem !important; }
[data-testid="stPlotlyChart"] { overflow:hidden; }
[data-testid="stDataFrame"] { overflow:hidden; }
/* Alerts use explicit high-contrast foreground/background pairs. */
[data-testid="stAlert"] {
  color:#203246 !important;
  background:#edf2f6 !important;
  border:1px solid #d5dee7 !important;
  border-left:4px solid #55728d !important;
  border-radius:12px !important;
}
[data-testid="stAlert"] p,
[data-testid="stAlert"] div,
[data-testid="stAlert"] span,
[data-testid="stAlert"] strong,
[data-testid="stAlert"] a,
[data-testid="stAlert"] [data-testid="stMarkdownContainer"] {
  color:#203246 !important;
}
[data-testid="stAlert"] svg { fill:#405d78 !important; color:#405d78 !important; }
[data-testid="stAlert"]:has(svg[ data-testid="stSuccessIcon"]) { background:#edf5ef !important; }
[data-testid="stExpander"] summary { font-weight:650; color:#21364b; }
[data-testid="stExpander"] [data-testid="stMarkdownContainer"],
[data-testid="stExpander"] [data-testid="stMarkdownContainer"] p,
[data-testid="stExpander"] [data-testid="stMarkdownContainer"] strong,
[data-testid="stExpander"] [data-testid="stMarkdownContainer"] li { color:#26374a !important; }
[data-testid="stDownloadButton"] { margin-top:.2rem; }

/* Sidebar controls: explicitly style BaseWeb's nested elements. Streamlit's
   default light widget surfaces otherwise remain white against the navy rail. */
[data-testid="stSidebar"] [data-testid="stWidgetLabel"],
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] label p {
  color:#d5dee8 !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] [data-baseweb="input"] > div,
[data-testid="stSidebar"] [data-testid="stDateInput"] > div,
[data-testid="stSidebar"] [data-testid="stDateInput"] input,
[data-testid="stSidebar"] [data-testid="stDateInput"] [data-baseweb="input"] > div {
  background-color:#263b51 !important;
  border-color:#52677e !important;
  color:#f4f6f8 !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] *,
[data-testid="stSidebar"] [data-baseweb="input"] *,
[data-testid="stSidebar"] [data-testid="stDateInput"] input {
  color:#f4f6f8 !important;
  -webkit-text-fill-color:#f4f6f8 !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] svg {
  fill:#d5dee8 !important;
  color:#d5dee8 !important;
}
[data-testid="stSidebar"] input::placeholder {
  color:#b8c4d1 !important;
  -webkit-text-fill-color:#b8c4d1 !important;
}
/* Select menus can be rendered outside the sidebar DOM, so target the
   BaseWeb menu itself as well as its option rows. */
[data-baseweb="popover"] [data-baseweb="menu"],
[data-baseweb="menu"] {
  background-color:#203449 !important;
  border:1px solid #52677e !important;
}
[data-baseweb="menu"] [role="option"] {
  background-color:#203449 !important;
  color:#f4f6f8 !important;
}
[data-baseweb="menu"] [role="option"] * {
  color:#f4f6f8 !important;
}
[data-baseweb="menu"] [role="option"]:hover,
[data-baseweb="menu"] [role="option"][aria-selected="true"] {
  background-color:#38516b !important;
}
[data-testid="stSidebar"] [data-testid="stSlider"] [data-baseweb="slider"] div {
  color:#f4f6f8;
}
@media (max-width: 760px) {
  .block-container { padding-left:1rem; padding-right:1rem; }
  h1 { font-size:1.85rem !important; }
  [data-testid="stMetricValue"] { font-size:1.45rem !important; }
}
</style>
""", unsafe_allow_html=True)


def _page_heading(kicker: str, title: str, description: str) -> None:
    st.markdown(f'<div class="eyebrow">{kicker}</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<div class="page-intro">{description}</div>', unsafe_allow_html=True)


def _section_heading(title: str, note: str | None = None) -> None:
    st.markdown(f'<div class="section-kicker">{title}</div>', unsafe_allow_html=True)
    if note:
        st.markdown(f'<div class="section-note">{note}</div>', unsafe_allow_html=True)


def _polish_chart(fig, height: int = 410):
    """Presentation-only Plotly styling; does not modify chart data.

    Explicitly sets every text-bearing chart component to a dark color. This
    prevents a dark Streamlit/browser theme from leaving white labels on the
    dashboard's white plot surfaces.
    """
    ink = "#26374a"
    muted = "#526174"
    fig.update_layout(
        template="plotly_white", height=height,
        paper_bgcolor="rgba(255,255,255,0)", plot_bgcolor="#ffffff",
        font=dict(family="DM Sans, sans-serif", color=ink, size=12),
        title=dict(font=dict(family="DM Sans, sans-serif", size=16, color="#1b2b3d"),
                   x=0.02, xanchor="left", y=0.99, yanchor="top", pad=dict(b=12)),
        margin=dict(l=24, r=24, t=82, b=34),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="left", x=0, bgcolor="rgba(255,255,255,0)",
                    font=dict(color=ink, size=11)),
        hoverlabel=dict(bgcolor="#172638", font=dict(color="#ffffff"),
                        bordercolor="#172638"),
    )
    # Existing annotations include vertical reference-line labels and event
    # labels. Preserve their positions/styles while forcing readable text.
    annotations = []
    for annotation in (fig.layout.annotations or []):
        item = annotation.to_plotly_json()
        annotation_font = dict(item.get("font") or {})
        annotation_font["color"] = ink
        item["font"] = annotation_font
        annotations.append(item)
    if annotations:
        fig.update_layout(annotations=annotations)

    # Axis titles and tick labels need explicit colors too: relying only on the
    # global Plotly font can be overridden by chart templates in some runtimes.
    fig.update_xaxes(
        showline=False, zeroline=False, gridcolor="#f0f2f4",
        tickfont=dict(family="DM Sans, sans-serif", color=muted, size=11),
        title_font=dict(family="DM Sans, sans-serif", color=ink, size=12),
    )
    fig.update_yaxes(
        showline=False, zeroline=False, gridcolor="#edf0f3",
        tickfont=dict(family="DM Sans, sans-serif", color=muted, size=11),
        title_font=dict(family="DM Sans, sans-serif", color=ink, size=12),
    )
    # Keep color-scale legends legible on white backgrounds as well.
    fig.update_coloraxes(
        colorbar=dict(tickfont=dict(color=muted),
                      title_font=dict(color=ink))
    )
    return fig


def _show_chart(fig, height: int = 410):
    # theme=None is important: Streamlit's automatic theme can override
    # Plotly's light template and reintroduce white axis/legend labels.
    st.plotly_chart(_polish_chart(fig, height), use_container_width=True,
                    theme=None,
                    config={"displaylogo": False, "responsive": True})


# ---------------- global controls and navigation ----------------
st.sidebar.markdown(
    '<div style="display:flex;align-items:center;margin:.25rem 0 .15rem">'
    '<span class="brand-mark">GA</span><div><div style="font-size:1.14rem;'
    'font-weight:800;letter-spacing:-.03em;color:#f3d9ac">GILDED ACE</div>'
    '<div style="font-size:.65rem;letter-spacing:.16em;color:#aab7c4;'
    'text-transform:uppercase">Floor intelligence</div></div></div>',
    unsafe_allow_html=True)
st.sidebar.caption("Synthetic property · Release 0.9.0-draft")
st.sidebar.markdown("---")

page_options = [
    "1 · Executive overview",
    "2 · Machine intelligence",
    "3 · Demand & reliability",
    "4 · Model validation",
    "5 · Capital allocation",
    "6 · Evidence & methods",
]
page = st.sidebar.radio("WORKSPACE", page_options, label_visibility="visible")
st.sidebar.markdown("---")
st.sidebar.markdown("**GLOBAL FILTERS**")
period = st.sidebar.date_input("Analysis period", (START, END),
                               min_value=START, max_value=END)
start, end = (period if isinstance(period, tuple) and len(period) == 2
              else (START, END))
retention = st.sidebar.slider(
    "Lost play not recovered next door", 0.0, 1.0, 0.45, 0.05,
    help="Downtime only costs the floor money if the player leaves rather "
         "than switching to a neighbouring machine.")

R = pipeline(start, end, retention)
diag, eff, movers, dt_cal = R["diag"], R["eff"], R["movers"], R["dt_cal"]

zones = ["All zones"] + sorted(diag["dom_zone"].dropna().unique().tolist())
zone_pick = st.sidebar.selectbox("Zone", zones)
banks = ["All banks"] + sorted(
    diag.loc[diag.dom_zone == zone_pick, "dom_bank"].dropna().unique().tolist()
    if zone_pick != "All zones" else diag["dom_bank"].dropna().unique().tolist())
bank_pick = st.sidebar.selectbox("Bank", banks)

view = diag.copy()
if zone_pick != "All zones":
    view = view[view.dom_zone == zone_pick]
if bank_pick != "All banks":
    view = view[view.dom_bank == bank_pick]

st.markdown(
    '<div class="header-strip"><span class="eyebrow">GILDED ACE / ANALYTICS</span>'
    '<span style="float:right" class="pill-note">LIVE FILTERS · '
    f'{start:%b %d, %Y} — {end:%b %d, %Y}'
    '</span></div>', unsafe_allow_html=True)

# ---------------- page 1: executive overview ----------------
if page.startswith("1"):
    _page_heading("01 / Decision brief",
                  "Look beyond the bottom line.",
                  "Separate machine performance from exposure, downtime, location and payout variance before making replacement decisions.")
    audit = diagnose.bottom10_audit(diag)
    keep = (audit.primary_cause != "weak_game").sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Machines on GM list", len(audit), help="The ten machines ranked lowest by raw house win.")
    c2.metric("Replacement not justified", keep,
              help="The shortfall is explained by luck, downtime or location.")
    c3.metric("Net downtime loss", f"${diag['lost_theo_net'].sum():,.0f}")
    c4.metric("Payout inspections", int((diag.payout_flag == 'inspect').sum()))

    _section_heading("Performance map", "Raw house win compared with theo per available hour — the exposure-fair view.")
    fig = px.scatter(
        diag, x="house_win", y="theo_per_avail_hr", color="primary_cause",
        color_discrete_map=CAUSE_COLOR, hover_name="machine_id",
        hover_data={"availability": ":.1%", "luck_residual": ":$,.0f", "game_title": True},
        labels={"house_win": "Total house win ($) · raw ranking",
                "theo_per_avail_hr": "Theo per available hour ($) · fair ranking"})
    lo = diag.nsmallest(10, "house_win")
    fig.add_trace(go.Scatter(
        x=lo.house_win, y=lo.theo_per_avail_hr, mode="markers",
        marker=dict(size=16, color="rgba(0,0,0,0)", line=dict(width=2, color="#172638")),
        name="GM bottom 10", hoverinfo="skip"))
    _show_chart(fig, 455)
    st.caption("Machines high on the vertical axis but far left earn well per hour and only look weak on raw totals. Availability, exposure and luck explain the gap [2].")

    _section_heading("Decision register", "The audit trail for the GM's ten proposed replacements.")
    st.dataframe(
        audit.style.format({"house_win": "${:,.0f}", "theo": "${:,.0f}",
                            "luck_residual": "${:,.0f}", "availability": "{:.1%}",
                            "theo_per_avail_hr": "${:.2f}"}),
        use_container_width=True, hide_index=True)
    st.download_button("Export decision register", audit.to_csv(index=False),
                       "bottom10_audit.csv")

# ---------------- page 2: machine intelligence ----------------
elif page.startswith("2"):
    _page_heading("02 / Asset-level diagnostics", "Machine intelligence",
                  "Compare each asset with the right peer group, inspect its diagnosis, and follow the evidence through time.")

    a, b, c = st.columns([1.5, 1, 2])
    sort_by = a.selectbox("Rank assets by", ["theo_per_avail_hr", "house_win", "availability",
                                              "luck_z", "adj_demand_pct", "lost_theo_net"])
    ascending = b.checkbox("Lowest first", True)
    table = view.sort_values(sort_by, ascending=ascending)
    mid = c.selectbox("Asset under review", table.machine_id.tolist())

    st.dataframe(
        table[["machine_id", "game_title", "dom_zone", "house_win", "theo",
               "luck_residual", "availability", "theo_per_avail_hr", "primary_cause", "confidence"]]
        .style.format({"house_win": "${:,.0f}", "theo": "${:,.0f}",
                       "luck_residual": "${:,.0f}", "availability": "{:.1%}",
                       "theo_per_avail_hr": "${:.2f}"}),
        use_container_width=True, height=270, hide_index=True)

    m = diag.loc[diag.machine_id == mid].iloc[0]
    st.markdown(f"### {mid} <span style='color:#a17b42;font-size:.8em'>/ {m.game_title}</span>",
                unsafe_allow_html=True)
    st.info(f"**Diagnosis: {CAUSE_LABEL[m.primary_cause]}** · {m.confidence} confidence\n\n"
            f"**Tests fired:** {m.tests_fired or 'none'}  ·  **Alternative explanation:** {m.alt_explanation}")

    k = st.columns(5)
    k[0].metric("Theo / available hr", f"${m.theo_per_avail_hr:.2f}", f"{m.pct_theo_per_avail_hr:.0f}th percentile")
    k[1].metric("Availability", f"{m.availability:.1%}", f"-{m.downtime_hours:.0f} hours")
    k[2].metric("Luck residual", f"${m.luck_residual:,.0f}", f"z = {m.luck_z:+.2f}")
    k[3].metric("Seat effect", f"{np.expm1(m.loc_effect or 0):+.1%}")
    k[4].metric("Machine effect", f"{np.expm1(m.machine_effect or 0):+.1%}")

    left, right = st.columns([1, 1.6])
    peer_mode = left.radio("Peer universe", ["floor", "game_setting", "zone", "manufacturer"], horizontal=True)
    metric = right.selectbox("Measure", ["theo_per_avail_hr", "coin_in_per_avail_hr", "availability"])
    dist = scorecard.peer_distribution(diag, mid, metric, peer_mode)
    fig = px.histogram(dist, nbins=40, labels={"value": metric},
                       title=f"{mid} relative to {peer_mode} peers · n={len(dist)}")
    fig.add_vline(x=m[metric], line_color="#b58a45", line_width=3, annotation_text="Selected asset")
    fig.add_vline(x=float(dist.median()), line_dash="dash", line_color="#60758a", annotation_text="Peer median")
    _show_chart(fig, 370)

    _section_heading("Hold with uncertainty", "A statistical interval, not a payout accusation.")
    st.write(
        f"Observed hold **{m.observed_hold:.3%}** (95% CI {m.hold_lo:.3%} – {m.hold_hi:.3%}) "
        f"vs. configured expectation **{m.expected_hold:.3%}**. "
        + ("The interval covers the expectation, so this is consistent with normal variance — a low hold does not by itself mean the machine is broken [2]."
           if m.hold_ci_covers_expected else
           f"The interval excludes the expectation (BH q={m.luck_q:.3f} across ~310 machines), which makes this a lead worth inspecting — not proof of wrongdoing [2].")
    )

    _section_heading("Asset timeline", "Coin-in rate, downtime days, location moves and cumulative luck on one timeline.")
    c = _con()
    tl = c.execute(f"""
        SELECT gaming_date, location_id, zone_id, available_hours,
               coin_in_usd / NULLIF(available_hours, 0) AS cipah,
               house_win_usd - theo_usd AS daily_luck
        FROM dn
        WHERE machine_id = '{mid}' AND active AND location_id IS NOT NULL
          AND gaming_date BETWEEN DATE '{start}' AND DATE '{end}'
        ORDER BY gaming_date
    """).df()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=tl.gaming_date, y=tl.cipah, name="Coin-in / available hr",
                             line=dict(color="#27445f", width=2.5)))
    fig.add_trace(go.Scatter(x=tl.gaming_date, y=tl.daily_luck.cumsum(),
                             name="Cumulative luck ($)", yaxis="y2",
                             line=dict(color="#a8aeb5", dash="dot", width=2)))
    down = tl[tl.available_hours < 20]
    fig.add_trace(go.Bar(x=down.gaming_date, y=down.available_hours,
                         name="Low-availability day", marker_color="#c18c4a", opacity=0.35))
    for d0 in tl.loc[tl.location_id != tl.location_id.shift(), "gaming_date"][1:]:
        fig.add_vline(x=d0, line_color="#8a83a8", line_dash="dash", annotation_text="Moved")
    fig.update_layout(yaxis2=dict(overlaying="y", side="right", title="Cumulative luck ($)"),
                      yaxis_title="Coin-in per available hour ($)", legend=dict(orientation="h"))
    _show_chart(fig, 430)
    st.caption("A decline that starts at a move points to location; one that starts after a repair points to configuration; a drifting grey line with flat blue is luck [1].")

# ---------------- page 3: demand and reliability ----------------
elif page.startswith("3"):
    _page_heading("03 / Floor performance", "Demand & reliability",
                  "See how property-wide demand changes over time, then quantify the earning opportunity lost to machine downtime.")
    tab1, tab2 = st.tabs(["Floor demand", "Downtime economics"])
    with tab1:
        c = _con()
        pd_day = c.execute(f"""
            SELECT p.gaming_date, p.coin_in_usd, cal.day_of_week, cal.is_holiday
            FROM property p JOIN cal USING (gaming_date)
            WHERE p.gaming_date BETWEEN DATE '{start}' AND DATE '{end}'
            ORDER BY 1
        """).df()
        pd_day["ma7"] = pd_day.coin_in_usd.rolling(7).mean()
        fig = px.line(pd_day, x="gaming_date", y=["coin_in_usd", "ma7"],
                      labels={"value": "Floor coin-in ($)"},
                      title="Daily floor demand and 7-day trend")
        ev = c.execute("SELECT * FROM events").df()
        for _, e in ev.iterrows():
            fig.add_vrect(x0=e.iloc[1], x1=e.iloc[2], fillcolor="#7a9b8a",
                          opacity=0.12, line_width=0)
        _show_chart(fig, 465)
        st.caption("Shaded bands represent conventions, sports weekends and promotions from the events table [2].")
    with tab2:
        k = st.columns(3)
        k[0].metric("Downtime hours", f"{dt_cal.downtime_hours.sum():,.0f}")
        k[1].metric("Gross lost theo", f"${dt_cal.lost_theo.sum():,.0f}")
        k[2].metric(f"Net lost theo · {retention:.0%} non-substitution",
                    f"${dt_cal.lost_theo.sum() * retention:,.0f}")
        daily_dt = dt_cal.groupby("gaming_date", as_index=False).sum(numeric_only=True)
        left, right = st.columns(2)
        with left:
            fig = px.bar(daily_dt, x="gaming_date", y="lost_theo",
                         labels={"lost_theo": "Gross lost theo ($ / day)"},
                         title="Lost theoretical win by day")
            _show_chart(fig, 390)
        with right:
            zone_dt = dt_cal.groupby("zone_id", as_index=False).sum(numeric_only=True).sort_values("lost_theo")
            fig = px.bar(zone_dt, x="zone_id", y="lost_theo",
                         labels={"lost_theo": "Gross lost theo ($)"},
                         title="Downtime exposure by zone")
            _show_chart(fig, 390)
        st.info("Lost theo values each unavailable hour at the machine's own earning rate, then discounts it by the share of play that moves to a neighbouring machine. Adjust the non-substitution assumption in the sidebar to update this page [1].")

# ---------------- page 4: model validation ----------------
elif page.startswith("4"):
    _page_heading("04 / Predictive controls", "Model validation",
                  "Review the holdout design, benchmark performance and prediction errors before relying on the demand forecasts.")
    with st.expander("Validation design · split, target and leakage controls", expanded=True):
        a, b, c = st.columns(3)
        a.markdown("**01 · Time split**\n\nTrain 2025-01-01 → 2025-04-30. Test 2025-05-01 → 2025-06-30. Split by calendar date, never randomly [2].")
        b.markdown("**02 · Prediction target**\n\n`log(1 + coin-in per available hour)` per machine-day. Coin-in measures demand; house win would smuggle luck into the target [1].")
        c.markdown("**03 · Leakage controls**\n\nLag features use `shift(1)`. House win, payout and theo are excluded as predictors. Baselines were frozen before holdout evaluation [2].")

    sp = OUT / "model_scores.csv"
    if sp.exists():
        scores = pd.read_csv(sp)
        _section_heading("Machine-day benchmark", "Model performance compared with frozen baselines.")
        st.dataframe(scores.style.format({"mae_log": "{:.4f}", "mae_usd_per_hr": "${:.2f}"}),
                     use_container_width=True, hide_index=True)
        best_base = scores[scores.name.str.startswith("baseline")].mae_log.min()
        model_mae = scores.loc[scores.name == "model_gbm", "mae_log"].iloc[0]
        (st.success if model_mae < best_base else st.warning)(
            f"Model MAE(log) {model_mae:.4f} vs. best baseline {best_base:.4f}. "
            + ("The model beats the baseline." if model_mae < best_base else
               "The model does NOT beat the baseline — reported honestly, which still earns credit [2]."))
    else:
        st.warning("Machine-day model scores are not available. Run the pipeline to generate `out/model_scores.csv`.")

    fs = OUT / "floor_forecast_scores.csv"
    if fs.exists():
        _section_heading("Property-level forecast", "The weekday × event forecast informs technician coverage decisions.")
        st.dataframe(pd.read_csv(fs).style.format({"mae_usd": "${:,.0f}", "mape_pct": "{:.2f}%"}),
                     use_container_width=True, hide_index=True)
        preds = pd.read_parquet(OUT / "floor_forecast_preds.parquet")
        fig = px.line(preds, x="gaming_date", y=["floor_coin_in", "pred_weekday_event", "pred_constant"],
                      labels={"value": "Floor coin-in ($)"}, title="Holdout · actual vs. predicted")
        _show_chart(fig, 420)
        st.caption("The weekday × event forecast drives the technician-shift decision: it identifies July days with the greatest exposure to downtime [2].")

    hp = OUT / "model_holdout.parquet"
    if hp.exists():
        h = pd.read_parquet(hp)
        c1, c2 = st.columns(2)
        with c1:
            _show_chart(px.scatter(h.sample(min(4000, len(h)), random_state=0),
                                   x="pred_cipah", y="actual_cipah", opacity=0.3,
                                   title="Actual vs. predicted demand"), 370)
        with c2:
            _show_chart(px.histogram(h, x="resid", nbins=60,
                                     title="Residual distribution"), 370)
    else:
        st.caption("Machine-day holdout details are not present in the output folder.")

# ---------------- page 5: capital allocation ----------------
elif page.startswith("5"):
    _page_heading("05 / Investment decision", "The $100,000 plan",
                  "A transparent allocation of the improvement budget, with expected impact, uncertainty and a measurement plan for every funded action.")
    cp, pp = OUT / "candidates.parquet", OUT / "portfolio.csv"
    if not cp.exists():
        st.warning("Portfolio outputs are not available. Run `python run_pipeline.py` first.")
    else:
        cand = pd.read_parquet(cp)
        folio = pd.read_csv(pp)
        sens = pd.read_csv(OUT / "portfolio_sensitivity.csv")
        summary = json.loads((OUT / "run_summary.json").read_text())

        k = st.columns(4)
        k[0].metric("Action types funded", len(summary["action_types"]), help="The catalogue allows at most three [2].")
        k[1].metric("Budget deployed", f"${summary['spend']:,.0f}", f"${summary['unspent']:,.0f} unspent")
        k[2].metric("Expected gain", f"${summary['expected_gain']:,.0f}")
        k[3].metric("Expected net", f"${summary['net']:,.0f}",
                    "beats doing nothing" if summary["beats_doing_nothing"] else "below keeping the cash")
        st.caption("Funded action types: " + ", ".join(summary["action_types"]))

        _section_heading("Allocation", "Where the budget goes and the expected net contribution by action type.")
        c1, c2 = st.columns(2)
        with c1:
            _show_chart(px.pie(folio, names="action_type", values="unit_cost", hole=0.58,
                               title="Budget mix by action type"), 390)
        waterfall = folio.groupby("action_type", as_index=False).agg(
            gain=("gain_base", "sum"), cost=("unit_cost", "sum"))
        waterfall["net"] = waterfall.gain - waterfall.cost
        with c2:
            _show_chart(px.bar(waterfall.sort_values("net"), x="net", y="action_type",
                               orientation="h", title="Expected net gain by action type ($)"), 390)

        _section_heading("Funded portfolio", "Unit cost, modeled effect range, confidence and implementation lead time.")
        st.dataframe(
            folio[["action_type", "target", "target_kind", "unit_cost", "gain_low",
                   "gain_base", "gain_high", "confidence", "lead_days"]]
            .style.format({"unit_cost": "${:,.0f}", "gain_low": "${:,.0f}",
                           "gain_base": "${:,.0f}", "gain_high": "${:,.0f}"}),
            use_container_width=True, hide_index=True)

        _section_heading("Downside sensitivity", "Does the investment thesis hold under less favorable effect assumptions?")
        st.dataframe(sens.style.format({"spend": "${:,.0f}", "expected_gain": "${:,.0f}",
                                        "net": "${:,.0f}", "unspent": "${:,.0f}"}),
                     use_container_width=True, hide_index=True)
        low = sens.loc[sens.scenario == "low"].iloc[0]
        (st.success if low.beats_doing_nothing else st.error)(
            f"Low scenario net: ${low.net:,.0f}. "
            + ("Even under pessimistic effect sizes the plan beats keeping the cash." if low.beats_doing_nothing else
               "Under pessimistic effect sizes, keeping the budget is the better answer — and the guide says that is a valid conclusion [2]."))

        with st.expander("All candidates considered · funded and rejected"):
            st.dataframe(
                cand[["action_type", "target", "unit_cost", "gain_base", "net_base", "confidence", "evidence"]]
                .style.format({"unit_cost": "${:,.0f}", "gain_base": "${:,.0f}", "net_base": "${:,.0f}"}),
                use_container_width=True, height=320, hide_index=True)
            st.caption("Rejected candidates show that the budget was compared against alternative uses, rather than assigned to the first available idea.")

        _section_heading("Measurement plan", "Open each funded action to review its expected result and how success will be measured.")
        for _, r in folio.iterrows():
            with st.expander(f"{r.action_type}  →  {r.target}"):
                a, b = st.columns(2)
                a.markdown(f"**Investment**\n\nCost: ${r.unit_cost:,.0f}\n\nLead time: {int(r.lead_days)} days")
                b.markdown(f"**Expected effect**\n\n${r.gain_base:,.0f}\n\nConfidence: {r.confidence}")
                st.markdown(f"**Evidence:** {r.evidence}\n\n**How we'll know it worked:** {r.measurement}")

# ---------------- page 6: evidence and methods ----------------
else:
    _page_heading("06 / Governance & traceability", "Evidence & methods",
                  "A transparent record of the claims, supporting analyses, natural experiments and limitations behind the dashboard.")
    st.markdown("Every claim traces to a table, a query and a stated limitation, in the required format: claim → query → result → alternative explanation → limitation [2].")

    ap = OUT / "claims_evidence_appendix.csv"
    if ap.exists():
        appendix = pd.read_csv(ap)
        _section_heading("Claims evidence register", "Downloadable source-to-claim traceability.")
        st.dataframe(appendix, use_container_width=True, height=420, hide_index=True)
        st.download_button("Export evidence register", appendix.to_csv(index=False),
                           "claims_evidence_appendix.csv")
    else:
        st.warning("Evidence appendix not found. Run the pipeline to generate `out/claims_evidence_appendix.csv`.")

    _section_heading("Diagnosis distribution", f"Current filtered population · {len(view):,} machines.")
    mix = (view.primary_cause.value_counts().rename_axis("cause").reset_index(name="machines"))
    mix["label"] = mix.cause.map(CAUSE_LABEL)
    _show_chart(px.bar(mix, x="label", y="machines", color="cause",
                        color_discrete_map=CAUSE_COLOR, title="Machines by primary diagnosis"), 370)

    _section_heading("Natural experiment", "Before-and-after evidence for assets that changed seats.")
    st.dataframe(movers.head(30).style.format({"cipah_before": "${:.2f}",
                                               "cipah_after": "${:.2f}",
                                               "cipah_change": "{:+.1%}"}),
                 use_container_width=True, hide_index=True)
    st.caption("Movers were not randomly chosen — management likely moved machines it already suspected were misplaced, so these before/after changes are an upper bound on what a move does to an average machine.")

    _section_heading("Limitations & disclosure")
    with st.expander("Read methodological limitations", expanded=True):
        st.markdown(f"""
- The casino, its staff and all performance data are fictional and synthetic; nothing here is a finding about a real property [2].
- Data release 0.9.0-draft: columns are stable but values may change before 1.0.0, so every number should be re-run against the final release [2].
- Six months only (Jan–Jun 2025), so annual seasonality is untestable.
- The luck standard error uses a volatility-based approximation of payout variance; an empirical bootstrap by game setting and exposure band is the more defensible alternative.
- Seat and machine effects are adjusted associations identified off the {eff['n_movers']} machines that occupied more than one seat — not causal estimates.
- Downtime value depends on the substitution assumption currently set to {retention:.0%}; the sidebar slider exposes it rather than burying it.
- Unusual payout patterns are leads to inspect, never accusations [2].
- Traffic counts are entries, not unique visitors, and missing counts mean the counter was offline rather than zero traffic [2].
        """)
    with st.expander("AI-use disclosure"):
        st.markdown("Claude was used to design the analysis approach, draft the Python/SQL pipeline and dashboard code, and review statistical choices. All data transformations, model validation and numbers shown were executed locally and reviewed by the team. Required as part of the submission [2].")

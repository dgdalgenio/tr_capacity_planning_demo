"""
Tarro Capacity Planning — RTA Console
--------------------------------------
A minimum-viable Streamlit app for a Real-Time Analyst (RTA) to:
  1) Review forecast vs. actual call volume, and auto-generate a reforecast
     for the remaining intervals of the day (model logic ported from
     1_forecast.ipynb).
  2) Translate the reforecast into required headcount per tier, using
     AHT / Utilization / Shrinkage assumptions (ported from 2_staffing.ipynb).
  3) Compare required headcount to actual agent readiness with a waterfall
     (tier-flex) allocation, flag deficits/surpluses, and surface
     recommended real-time adjustments.

Forecast & actual call volume are editable (type or paste values) so the
RTA can update the plan as the day unfolds.
"""

import io
import math
from datetime import datetime, time as dtime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sklearn.linear_model import LinearRegression, Ridge

try:
    from statsmodels.tsa.holtwinters import Holt
    from statsmodels.tsa.arima.model import ARIMA
    HAVE_STATSMODELS = True
except Exception:
    HAVE_STATSMODELS = False

st.set_page_config(page_title="Tarro Capacity Planning — RTA Console", layout="wide")

DATA_DIR = "data"
TIERS = ["Tier 1", "Tier 2", "Tier 3"]
INTERVAL_SECONDS = 900  # 15-minute intervals
DECAY_RATE_DEFAULT = 0.85

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def _to_time_str(x):
    """Normalize a variety of time inputs to HH:MM:SS string."""
    if isinstance(x, dtime):
        return x.strftime("%H:%M:%S")
    s = str(x).strip()
    for fmt in ("%H:%M:%S", "%H:%M", "%I:%M %p", "%I:%M:%S %p"):
        try:
            return datetime.strptime(s, fmt).strftime("%H:%M:%S")
        except ValueError:
            continue
    return s


@st.cache_data
def load_default_forecast_actual():
    df = pd.read_csv(f"{DATA_DIR}/forecast_actual.csv")
    for c in df.columns:
        if c != "Time":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


@st.cache_data
def load_default_weekly_avg():
    return pd.read_csv(f"{DATA_DIR}/weekly_avg.csv")


@st.cache_data
def load_default_staffing_ready():
    return pd.read_csv(f"{DATA_DIR}/staffing_ready.csv")


def build_model(model_type):
    if model_type == "linear":
        return LinearRegression()
    elif model_type == "ridge":
        return Ridge(alpha=1.0)
    raise ValueError(model_type)


def apply_decay(diff_series, decay_rate):
    steps = np.arange(1, len(diff_series) + 1)
    return diff_series * (decay_rate ** steps)


def generate_reforecast(fc_df, ac_df, tiers, decay_rate):
    """
    fc_df, ac_df: DataFrames indexed by interval order, columns = tiers.
    Returns (reforecast_df, diagnostics dict) for the FUTURE rows only.
    Ported from 1_forecast.ipynb (model selection + decayed-diff reforecast).
    """
    has_actuals = ac_df.notna().all(axis=1) & (ac_df.abs().sum(axis=1) > 0)
    known_idx = ac_df.index[has_actuals]
    future_idx = ac_df.index.difference(known_idx, sort=False)

    diagnostics = {"best_model": {}, "error": {}, "n_known": len(known_idx)}

    if len(known_idx) < 4 or len(future_idx) == 0:
        # Not enough history to model drift — just carry the forecast forward.
        return fc_df.loc[future_idx, tiers].copy(), diagnostics

    actual_known = ac_df.loc[known_idx, tiers]
    forecast_known = fc_df.loc[known_idx, tiers]
    diff_known = actual_known - forecast_known

    n = len(known_idx)
    x = np.arange(n).reshape(-1, 1)
    x_train, x_last = x[:-1], x[-1:]
    x_future = np.arange(n, n + len(future_idx)).reshape(-1, 1)

    model_types = ["linear", "ridge"]
    if HAVE_STATSMODELS and n >= 6:
        model_types += ["holt", "arima"]

    candidates = {}
    for mt in model_types:
        try:
            if mt in ("linear", "ridge"):
                preds_last = {}
                for col in tiers:
                    m = build_model(mt)
                    m.fit(x_train, diff_known[col].values[:-1])
                    preds_last[col] = m.predict(x_last)[0]
                candidates[mt] = preds_last
            elif mt == "holt":
                preds_last = {}
                for col in tiers:
                    series = diff_known[col].values[:-1].astype(float)
                    m = Holt(series, initialization_method="estimated").fit(optimized=True)
                    preds_last[col] = m.forecast(1)[0]
                candidates[mt] = preds_last
            elif mt == "arima":
                preds_last = {}
                for col in tiers:
                    series = diff_known[col].values[:-1].astype(float)
                    m = ARIMA(series, order=(1, 1, 0)).fit()
                    preds_last[col] = m.forecast(1)[0]
                candidates[mt] = preds_last
        except Exception:
            continue

    actual_last_diff = {col: diff_known[col].values[-1] for col in tiers}
    best_model_per_tier = {}
    for col in tiers:
        best_mt, best_err = None, np.inf
        for mt, preds in candidates.items():
            err = abs(preds[col] - actual_last_diff[col])
            if err < best_err:
                best_mt, best_err = mt, err
        best_model_per_tier[col] = best_mt or "linear"
        diagnostics["best_model"][col] = best_model_per_tier[col]
        diagnostics["error"][col] = round(best_err, 3) if best_mt else None

    predicted_diff = pd.DataFrame(index=future_idx, columns=tiers, dtype=float)
    for col in tiers:
        mt = best_model_per_tier[col]
        series_full = diff_known[col].values.astype(float)
        try:
            if mt in ("linear", "ridge"):
                m = build_model(mt)
                m.fit(x, series_full)
                preds = m.predict(x_future)
            elif mt == "holt":
                m = Holt(series_full, initialization_method="estimated").fit(optimized=True)
                preds = m.forecast(len(future_idx))
            elif mt == "arima":
                m = ARIMA(series_full, order=(1, 1, 0)).fit()
                preds = m.forecast(len(future_idx))
        except Exception:
            m = build_model("linear")
            m.fit(x, series_full)
            preds = m.predict(x_future)
        predicted_diff[col] = preds

    predicted_diff_damped = predicted_diff.copy()
    for col in tiers:
        predicted_diff_damped[col] = apply_decay(predicted_diff[col].values, decay_rate)

    reforecast = (fc_df.loc[future_idx, tiers] + predicted_diff_damped).clip(lower=0)
    reforecast = reforecast.fillna(fc_df.loc[future_idx, tiers]).fillna(0)
    reforecast = reforecast.round(0).astype(int)
    return reforecast, diagnostics


def headcount_required(call_volume, aht, utilization, shrinkage, interval_seconds=INTERVAL_SECONDS):
    util = utilization.replace(0, np.nan)
    shrink = shrinkage.clip(upper=0.95)
    return (call_volume * aht) / (interval_seconds * util * (1 - shrink))


def waterfall_gap(req, ready):
    """Tier-flex waterfall: Tier 3 agents can flex down to serve Tier 2/1;
    Tier 2 agents can flex down to serve Tier 1. Ported from 2_staffing.ipynb."""
    t1_req, t2_req, t3_req = req["Tier 1"], req["Tier 2"], req["Tier 3"]
    t1_avail, t2_avail, t3_avail = ready["Tier 1"], ready["Tier 2"], ready["Tier 3"]

    t3_served = min(t3_req, t3_avail)
    t3_unmet = t3_req - t3_served
    t3_leftover = t3_avail - t3_served

    t2_capacity = t2_avail + t3_leftover
    t2_served = min(t2_req, t2_capacity)
    t2_unmet = t2_req - t2_served
    t2_leftover = t2_capacity - t2_served

    t1_capacity = t1_avail + t2_leftover
    t1_served = min(t1_req, t1_capacity)
    t1_unmet = t1_req - t1_served
    t1_leftover = t1_capacity - t1_served

    return {
        "Tier 1 Required": t1_req, "Tier 1 Ready": t1_avail, "Tier 1 Served": round(t1_served, 2), "Tier 1 Unmet": round(t1_unmet, 2),
        "Tier 2 Required": t2_req, "Tier 2 Ready": t2_avail, "Tier 2 Served": round(t2_served, 2), "Tier 2 Unmet": round(t2_unmet, 2),
        "Tier 3 Required": t3_req, "Tier 3 Ready": t3_avail, "Tier 3 Served": round(t3_served, 2), "Tier 3 Unmet": round(t3_unmet, 2),
        "Total Required": round(t1_req + t2_req + t3_req, 2),
        "Total Ready": round(t1_avail + t2_avail + t3_avail, 2),
        "Total Unmet": round(t1_unmet + t2_unmet + t3_unmet, 2),
        "Overall Surplus": round(t1_leftover, 2),
    }


# ----------------------------------------------------------------------------
# Sidebar — global assumptions
# ----------------------------------------------------------------------------

st.sidebar.title("⚙️ RTA Controls")
st.sidebar.caption("Friday, Oct 31, 2025 · Voice Platform · 40-agent team")

decay_rate = st.sidebar.slider(
    "Reforecast decay rate", 0.50, 0.99, DECAY_RATE_DEFAULT, 0.01,
    help="How fast the model-predicted forecast drift fades back to baseline over the remaining day. "
         "Higher = corrections persist longer.",
)

st.sidebar.markdown("---")
st.sidebar.subheader("Total agent headcount")
total_agents = st.sidebar.number_input("Agents on the floor today", min_value=1, value=40, step=1)

st.sidebar.markdown("---")
st.sidebar.subheader("Shift assumptions (fallback)")
st.sidebar.caption("Used only for intervals missing a weekly AHT/Util/Shrinkage estimate.")
fallback_aht = st.sidebar.number_input("Fallback AHT (seconds)", value=90, step=5)
fallback_util = st.sidebar.slider("Fallback Utilization Rate", 0.4, 1.0, 0.80, 0.01)
fallback_shrink = st.sidebar.slider("Fallback Shrinkage", 0.0, 0.6, 0.12, 0.01)

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Reset all tables to defaults"):
    st.cache_data.clear()
    for k in ["fc_ac_editor", "weekly_editor", "ready_editor"]:
        st.session_state.pop(k, None)
    st.rerun()

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------

st.title("📞 Tarro Capacity Planning — RTA Console")
st.caption(
    "Close the gap between plan and reality. Protect **customer experience**, "
    "**cost & efficiency**, and **agent quality of life** — in real time."
)

tab1, tab2, tab3 = st.tabs(
    ["1️⃣ Forecast vs. Actual & Reforecast", "2️⃣ Headcount Requirements", "3️⃣ Real-Time Staffing & Gaps"]
)

# ----------------------------------------------------------------------------
# TAB 1 — Forecast / Actual / Reforecast
# ----------------------------------------------------------------------------

with tab1:
    st.subheader("Call volume: forecast vs. actual")
    st.markdown(
        "Edit the table below directly (double-click a cell, then press **Tab/Enter** to commit) "
        "or **paste** values in from Excel. Leave *Actual* cells blank for intervals that haven't "
        "happened yet — the app regenerates the **reforecast** for those rows the moment you commit "
        "an edit. *Total* columns are auto-computed from the tier columns, so you never need to "
        "edit them by hand."
    )

    if "fc_ac_editor" not in st.session_state:
        st.session_state["fc_ac_editor"] = load_default_forecast_actual()

    edited_raw = st.data_editor(
        st.session_state["fc_ac_editor"],
        num_rows="fixed",
        use_container_width=True,
        height=420,
        column_config={
            "Time": st.column_config.TextColumn("Time Interval", disabled=True),
            "Fc_Total": st.column_config.NumberColumn("Forecast Total (auto)", disabled=True),
            "Fc_T1": st.column_config.NumberColumn("Forecast T1"),
            "Fc_T2": st.column_config.NumberColumn("Forecast T2"),
            "Fc_T3": st.column_config.NumberColumn("Forecast T3"),
            "Ac_Total": st.column_config.NumberColumn("Actual Total (auto)", disabled=True),
            "Ac_T1": st.column_config.NumberColumn("Actual T1"),
            "Ac_T2": st.column_config.NumberColumn("Actual T2"),
            "Ac_T3": st.column_config.NumberColumn("Actual T3"),
        },
        key="fc_ac_data_editor",
    )

    # Auto-sync the Total columns from whatever tier values were just edited, so the table
    # (and every downstream calculation) always reflects the latest inputs.
    edited = edited_raw.copy()
    edited["Fc_Total"] = edited[["Fc_T1", "Fc_T2", "Fc_T3"]].sum(axis=1, min_count=1)
    edited["Ac_Total"] = edited[["Ac_T1", "Ac_T2", "Ac_T3"]].sum(axis=1, min_count=1)

    # Detect whether this rerun was triggered by an actual edit to the volume table, so we can
    # surface a clear confirmation that the reforecast just updated.
    prev_signature = st.session_state.get("fc_ac_signature")
    new_signature = pd.util.hash_pandas_object(edited.fillna(-1)).sum()
    volume_changed = prev_signature is not None and prev_signature != new_signature
    st.session_state["fc_ac_signature"] = new_signature
    st.session_state["fc_ac_editor"] = edited

    fc_df = edited.set_index("Time")[["Fc_T1", "Fc_T2", "Fc_T3"]].rename(
        columns={"Fc_T1": "Tier 1", "Fc_T2": "Tier 2", "Fc_T3": "Tier 3"}
    )
    ac_df = edited.set_index("Time")[["Ac_T1", "Ac_T2", "Ac_T3"]].rename(
        columns={"Ac_T1": "Tier 1", "Ac_T2": "Tier 2", "Ac_T3": "Tier 3"}
    )

    reforecast_df, diag = generate_reforecast(fc_df, ac_df, TIERS, decay_rate)

    if volume_changed:
        st.toast("Reforecast updated from your latest edits ✅", icon="🔄")

    known_mask = ac_df.notna().all(axis=1) & (ac_df.abs().sum(axis=1) > 0)
    n_known, n_future = known_mask.sum(), (~known_mask).sum()

    c1, c2, c3 = st.columns(3)
    c1.metric("Intervals with actuals", int(n_known))
    c2.metric("Intervals reforecasted", int(n_future))
    c3.metric("Decay rate", f"{decay_rate:.2f}")

    if diag.get("best_model"):
        st.caption(
            "Best-fit drift model per tier (lowest backtest error): "
            + " · ".join(f"**{t}** → {diag['best_model'][t]} (err={diag['error'][t]})" for t in TIERS)
        )

    colors = {"Tier 1": "#2563eb", "Tier 2": "#f59e0b", "Tier 3": "#16a34a"}

    st.markdown("---")
    st.markdown("### 🔎 Forecast Model Analysis — current model accuracy")
    st.caption(
        "How well is today's original forecast tracking reality so far? This section only looks at "
        "intervals where actuals have already landed (no reforecast)."
    )

    if n_known == 0:
        st.info("No actuals entered yet — analysis will populate once at least one interval has actuals.")
    else:
        known_idx = fc_df.index[known_mask]
        fc_known = fc_df.loc[known_mask]
        ac_known = ac_df.loc[known_mask]

        # --- Actual vs Forecasted Tiers (known intervals only, no reforecast) ---
        fig_av = go.Figure()
        for col in TIERS:
            fig_av.add_trace(go.Scatter(x=fc_df.index, y=fc_df[col], name=f"{col} Forecast",
                                         line=dict(color=colors[col], dash="dot", width=2)))
            fig_av.add_trace(go.Scatter(x=ac_known.index, y=ac_known[col], name=f"{col} Actual",
                                         line=dict(color=colors[col], width=3)))
        fig_av.update_layout(
            title="Actual vs. Forecasted Tiers", xaxis_title="Time Interval", yaxis_title="Call Volume",
            height=440, legend=dict(orientation="h", yanchor="bottom", y=1.02), margin=dict(t=60),
        )
        st.plotly_chart(fig_av, use_container_width=True)

        # --- Forecast variance summary (per-interval diff & pct, known intervals) ---
        st.markdown("**Forecast variance summary** — per-interval difference between actual and forecast")
        variance_rows = pd.DataFrame(index=known_idx)
        all_cols = TIERS + ["Total"]
        fc_known_full = fc_known.copy()
        ac_known_full = ac_known.copy()
        fc_known_full["Total"] = fc_known[TIERS].sum(axis=1)
        ac_known_full["Total"] = ac_known[TIERS].sum(axis=1)
        for col in all_cols:
            diff = ac_known_full[col] - fc_known_full[col]
            pct = (diff / fc_known_full[col].replace(0, np.nan) * 100).round(1)
            variance_rows[f"{col} Diff"] = diff
            variance_rows[f"{col} Pct Var"] = pct
        st.dataframe(variance_rows, use_container_width=True)

        # --- Cumulative variance summary (per tier + Total) ---
        st.markdown("**Cumulative variance summary** — running totals for the intervals observed so far")
        rows = []
        for col in all_cols:
            fc_sum = fc_known_full[col].sum()
            ac_sum = ac_known_full[col].sum()
            diff = ac_sum - fc_sum
            pct = (diff / fc_sum * 100) if fc_sum else np.nan
            rows.append([col, fc_sum, ac_sum, diff, round(pct, 1)])
        summary_df = pd.DataFrame(rows, columns=["Segment", "Forecast Total", "Actual Total", "Diff", "Pct Variance"])
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        # --- Difference of Actual to Forecasted values (diff over time) ---
        fig_diff = go.Figure()
        fig_diff.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.6)
        diff_colors = {"Tier 1": colors["Tier 1"], "Tier 2": colors["Tier 2"], "Tier 3": colors["Tier 3"], "Total": "#6b21a8"}
        for col in all_cols:
            diff_series = ac_known_full[col] - fc_known_full[col]
            fig_diff.add_trace(go.Scatter(x=known_idx, y=diff_series, name=f"{col} Diff",
                                           line=dict(color=diff_colors[col], width=2)))
        fig_diff.update_layout(
            title="Difference of Actual to Forecasted Values", xaxis_title="Time Interval",
            yaxis_title="Actual − Forecast", height=420,
            legend=dict(orientation="h", yanchor="bottom", y=1.02), margin=dict(t=60),
        )
        st.plotly_chart(fig_diff, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🔮 Reforecast — remaining intervals")
    st.caption(
        "Projects the drift between actual and forecast forward for the intervals still ahead, using "
        "whichever model (Linear, Ridge, Holt, or ARIMA) best matched the most recent known drift per tier."
    )

    # Build combined series for plotting: actual where known, reforecast where future
    plot_df = pd.DataFrame(index=fc_df.index)
    for col in TIERS:
        plot_df[f"{col} Forecast"] = fc_df[col]
        actual_col = ac_df[col].copy()
        plot_df[f"{col} Actual"] = actual_col.where(known_mask)
        reforecast_col = pd.Series(index=fc_df.index, dtype=float)
        reforecast_col.loc[reforecast_df.index] = reforecast_df[col].values
        plot_df[f"{col} Reforecast"] = reforecast_col

    fig = go.Figure()
    for col in TIERS:
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df[f"{col} Forecast"], name=f"{col} Forecast",
                                  line=dict(color=colors[col], dash="dot", width=2)))
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df[f"{col} Actual"], name=f"{col} Actual",
                                  line=dict(color=colors[col], width=3)))
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df[f"{col} Reforecast"], name=f"{col} Reforecast",
                                  line=dict(color=colors[col], dash="dash", width=2), mode="lines+markers",
                                  marker=dict(size=4)))
    fig.update_layout(
        # title="Actual vs. Forecasted vs. Reforecasted Call Volume by Tier",
        xaxis_title="Time Interval", yaxis_title="Call Volume",
        height=480, legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=60),
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📋 Reforecast table (remaining intervals)"):
        rf_display = reforecast_df.copy()
        rf_display["Total"] = rf_display[TIERS].sum(axis=1)
        st.dataframe(rf_display, use_container_width=True)

    # Combined "final" call volume table used downstream: actuals where known, reforecast where future
    final_volume = pd.DataFrame(index=fc_df.index, columns=TIERS, dtype=float)
    final_volume.loc[known_mask, TIERS] = ac_df.loc[known_mask, TIERS].values
    final_volume.loc[reforecast_df.index, TIERS] = reforecast_df.values
    final_volume["Total"] = final_volume[TIERS].sum(axis=1)
    st.session_state["final_volume"] = final_volume

# ----------------------------------------------------------------------------
# TAB 2 — Headcount requirements
# ----------------------------------------------------------------------------

with tab2:
    st.subheader("Translate call volume into required headcount")
    st.markdown(
        r"$\text{Headcount Required} = \dfrac{\text{Call Volume} \times \text{AHT}}"
        r"{\text{Interval Seconds} \times \text{Utilization} \times (1 - \text{Shrinkage})}$"
    )
    st.caption(
        "Default AHT / Utilization / Shrinkage below are decay-weighted averages of the last 8 weeks "
        "of historical performance per interval (editable)."
    )

    if "weekly_editor" not in st.session_state:
        st.session_state["weekly_editor"] = load_default_weekly_avg()

    weekly_edit = st.data_editor(
        st.session_state["weekly_editor"],
        use_container_width=True,
        height=320,
        column_config={
            "Time": st.column_config.TextColumn("Time Interval", disabled=True),
            "Shrinkage": st.column_config.NumberColumn("Shrinkage (0-1)", format="%.3f"),
            "Average Handle Time": st.column_config.NumberColumn("AHT (sec)", format="%.1f"),
            "Utilization Rate": st.column_config.NumberColumn("Utilization (0-1)", format="%.3f"),
        },
        key="weekly_data_editor",
    )
    st.session_state["weekly_editor"] = weekly_edit

    weekly_edit = weekly_edit.copy()
    weekly_edit["Time"] = weekly_edit["Time"].apply(_to_time_str)
    weekly_indexed = weekly_edit.set_index("Time")

    final_volume = st.session_state.get("final_volume")
    if final_volume is None or final_volume.empty:
        st.info("Fill in Tab 1 first to compute headcount requirements.")
    else:
        vol_index = [_to_time_str(t) for t in final_volume.index]
        weekly_aligned = weekly_indexed.reindex(vol_index)
        weekly_aligned["Average Handle Time"] = weekly_aligned["Average Handle Time"].fillna(fallback_aht)
        weekly_aligned["Utilization Rate"] = weekly_aligned["Utilization Rate"].fillna(fallback_util)
        weekly_aligned["Shrinkage"] = weekly_aligned["Shrinkage"].fillna(fallback_shrink)
        weekly_aligned.index = final_volume.index

        missing = weekly_aligned[weekly_aligned.isna().any(axis=1)]
        if len(missing) > 0:
            st.warning(f"{len(missing)} interval(s) used fallback assumptions (no matching weekly data).")

        hc_required = pd.DataFrame(index=final_volume.index)
        for col in TIERS:
            hc_required[col] = headcount_required(
                final_volume[col].fillna(0),
                weekly_aligned["Average Handle Time"],
                weekly_aligned["Utilization Rate"],
                weekly_aligned["Shrinkage"],
            ).apply(lambda v: math.ceil(v) if pd.notna(v) else 0)
        hc_required["Total"] = hc_required[TIERS].sum(axis=1)
        st.session_state["hc_required"] = hc_required

        c1, c2, c3 = st.columns(3)
        c1.metric("Peak headcount required", int(hc_required["Total"].max()))
        c2.metric("Peak interval", str(hc_required["Total"].idxmax()))
        c3.metric("Avg headcount required", f"{hc_required['Total'].mean():.1f}")

        fig2 = go.Figure()
        for col in TIERS:
            fig2.add_trace(go.Scatter(x=hc_required.index, y=hc_required[col], name=f"{col} HC Required",
                                       stackgroup="one", line=dict(color=colors[col])))
        fig2.add_hline(y=total_agents, line_dash="dash", line_color="red",
                        annotation_text=f"Total agents on floor ({total_agents})", annotation_position="top left")
        fig2.update_layout(title="Required Headcount by Tier (stacked) vs. Total Agents on Floor",
                           xaxis_title="Time Interval", yaxis_title="Agents Required", height=440)
        st.plotly_chart(fig2, use_container_width=True)

        with st.expander("📋 Headcount required table"):
            st.dataframe(hc_required, use_container_width=True)

# ----------------------------------------------------------------------------
# TAB 3 — Real-time staffing & gaps
# ----------------------------------------------------------------------------

with tab3:
    st.subheader("Compare requirements to actual agent readiness")
    st.markdown(
        "Enter how many agents are **ready/available** per tier, per interval (default below is today's "
        "published schedule). The app applies a **tier-flex waterfall**: Tier 3 agents can flex down to "
        "cover Tier 2 or Tier 1 demand, and Tier 2 agents can flex down to cover Tier 1 — never the reverse."
    )

    if "ready_editor" not in st.session_state:
        st.session_state["ready_editor"] = load_default_staffing_ready()

    ready_edit = st.data_editor(
        st.session_state["ready_editor"],
        use_container_width=True,
        height=320,
        column_config={
            "Time": st.column_config.TextColumn("Time Interval", disabled=True),
            "Tier1_Ready": st.column_config.NumberColumn("Tier 1 Ready"),
            "Tier2_Ready": st.column_config.NumberColumn("Tier 2 Ready"),
            "Tier3_Ready": st.column_config.NumberColumn("Tier 3 Ready"),
        },
        key="ready_data_editor",
    )
    st.session_state["ready_editor"] = ready_edit

    ready_edit = ready_edit.copy()
    ready_edit["Time"] = ready_edit["Time"].apply(_to_time_str)
    ready_indexed = ready_edit.set_index("Time").rename(
        columns={"Tier1_Ready": "Tier 1", "Tier2_Ready": "Tier 2", "Tier3_Ready": "Tier 3"}
    )

    hc_required = st.session_state.get("hc_required")
    if hc_required is None:
        st.info("Complete Tab 1 and Tab 2 first to run the staffing gap analysis.")
    else:
        vol_index = [_to_time_str(t) for t in hc_required.index]
        ready_aligned = ready_indexed.reindex(vol_index).fillna(0)
        ready_aligned.index = hc_required.index

        scale_note = ""
        current_total_ready_peak = ready_aligned[TIERS].sum(axis=1).max()
        if current_total_ready_peak > 0 and abs(current_total_ready_peak - total_agents) > 0.5:
            scale = total_agents / current_total_ready_peak
            ready_aligned = (ready_aligned * scale).round(0)
            scale_note = (
                f"⚖️ Scaled the readiness schedule so its peak headcount matches your **{total_agents}**-agent "
                f"team (was peaking at {int(current_total_ready_peak)})."
            )

        results = []
        for idx in hc_required.index:
            r = waterfall_gap(hc_required.loc[idx, TIERS], ready_aligned.loc[idx, TIERS])
            r["Time"] = idx
            results.append(r)
        waterfall_df = pd.DataFrame(results).set_index("Time")
        st.session_state["waterfall_df"] = waterfall_df

        if scale_note:
            st.caption(scale_note)

        deficit_intervals = waterfall_df[waterfall_df["Total Unmet"] > 0]
        surplus_peak = waterfall_df["Overall Surplus"].max()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Intervals with a deficit", len(deficit_intervals),
                  delta=f"{len(deficit_intervals)/len(waterfall_df)*100:.0f}% of day" if len(waterfall_df) else None,
                  delta_color="inverse")
        c2.metric("Worst single-interval deficit", int(waterfall_df["Total Unmet"].max()))
        c3.metric("Peak overall surplus", int(surplus_peak))
        c4.metric("Total unmet-agent-intervals (day)", int(waterfall_df["Total Unmet"].sum()))

        fig3 = go.Figure()
        fig3.add_trace(go.Bar(x=waterfall_df.index, y=waterfall_df["Total Required"], name="Total Required",
                               marker_color="rgba(37,99,235,0.35)"))
        fig3.add_trace(go.Scatter(x=waterfall_df.index, y=waterfall_df["Total Ready"], name="Total Ready",
                                   line=dict(color="#16a34a", width=3)))
        fig3.add_trace(go.Scatter(x=waterfall_df.index, y=waterfall_df["Total Unmet"], name="Total Unmet",
                                   line=dict(color="#dc2626", width=2, dash="dot"), fill="tozeroy",
                                   fillcolor="rgba(220,38,38,0.15)"))
        fig3.update_layout(title="Required vs. Ready Headcount, with Unmet Gap Highlighted",
                           xaxis_title="Time Interval", yaxis_title="Agents", height=460,
                           legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig3, use_container_width=True)

        st.markdown("#### 🚨 Real-time adjustment recommendations")
        if len(deficit_intervals) == 0:
            st.success(
                "No coverage gaps detected under the tier-flex waterfall. Current staffing plan protects "
                "customer experience across all intervals — hold the plan and monitor actuals as they land."
            )
        else:
            worst = deficit_intervals.sort_values("Total Unmet", ascending=False).head(5)
            st.warning(
                f"**{len(deficit_intervals)} interval(s)** show an unmet headcount gap. Highest-risk windows:"
            )
            for t, row in worst.iterrows():
                tier_gaps = [f"{tier} short {int(row[f'{tier} Unmet'])}" for tier in TIERS if row[f"{tier} Unmet"] > 0]
                st.markdown(
                    f"- **{t}** — total unmet: **{row['Total Unmet']:.0f} agent(s)** "
                    f"({', '.join(tier_gaps)}). Consider pulling agents from break/aux, offering "
                    f"overtime/voluntary time-on, or flexing trained cross-skill agents into this window."
                )
            st.caption(
                "Prioritize plugging Tier 1 gaps first — this tier typically carries the highest call volume "
                "and the most latency-sensitive customer interactions."
            )

        with st.expander("📋 Full waterfall gap table"):
            st.dataframe(waterfall_df, use_container_width=True)

        csv_buf = io.StringIO()
        waterfall_df.to_csv(csv_buf)
        st.download_button(
            "⬇️ Download real-time staffing plan (CSV)",
            data=csv_buf.getvalue(),
            file_name="realtime_staffing_plan.csv",
            mime="text/csv",
        )

st.markdown("---")
st.caption(
    "MVP build · logic ported from 1_forecast.ipynb (reforecast modeling) and 2_staffing.ipynb "
    "(headcount + tier-flex waterfall) · edit any table above to reflect the latest actuals."
)
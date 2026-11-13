import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# streamlit puts this file's folder on the path, not the project root, so add it before importing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from prognosis.config import get_settings  # noqa: E402
from prognosis.ui.client import Api, ApiError  # noqa: E402

BAND_COLOR = {"low": "#2e7d32", "moderate": "#e68c00", "high": "#c62828"}

st.set_page_config(page_title="Prognosis", layout="wide")


def api() -> Api:
    if "api" not in st.session_state:
        st.session_state.api = Api(get_settings().api_url)
    return st.session_state.api


def show_errors(e: ApiError):
    for m in e.messages():
        st.error(m)


def login_screen():
    st.title("Prognosis")
    st.caption("Clinical risk decision support")
    with st.form("login"):
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        if st.form_submit_button("Sign in"):
            try:
                api().login(u, p)
                st.session_state.me = api().me()
                st.rerun()
            except ApiError as e:
                st.error("Could not sign in: " + e.messages()[0])
            except Exception:
                st.error("The API is not reachable. Is the server running?")


def gauge(prob, thresholds):
    lo, hi = thresholds["low_moderate"], thresholds["moderate_high"]
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=prob * 100, number={"suffix": "%", "valueformat": ".1f"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#222", "thickness": 0.25},
               "steps": [{"range": [0, lo * 100], "color": "#c8e6c9"},
                         {"range": [lo * 100, hi * 100], "color": "#ffe0b2"},
                         {"range": [hi * 100, 100], "color": "#ffcdd2"}]}))
    fig.update_layout(height=230, margin=dict(l=20, r=20, t=20, b=0))
    return fig


def contribution_chart(res):
    rows = [c for c in res["contributions"] if not c["imputed"] and abs(c["effect"]) >= 0.0005][:10][::-1]
    labels = [f"{c['label']}: " + (f"{c['value']:g}" if isinstance(c["value"], float) else str(c["value"])) for c in rows]
    fig = go.Figure(go.Bar(x=[c["effect"] * 100 for c in rows], y=labels, orientation="h",
                           marker_color=["#c62828" if c["effect"] > 0 else "#1e64b4" for c in rows]))
    fig.update_layout(height=60 + 34 * len(rows), margin=dict(l=10, r=10, t=10, b=30),
                      xaxis_title="change in probability (percentage points)")
    return fig


def feature_form(spec, key):
    inputs = {}
    cols = st.columns(3)
    for i, f in enumerate(spec["features"]):
        col = cols[i % 3]
        label = f["label"] + (f" ({f['unit']})" if f["unit"] and f["kind"] == "number" else "") + ("" if f["required"] else " (optional)")
        k = f"{key}_{f['name']}"
        if f["kind"] == "number":
            inputs[f["name"]] = col.number_input(label, min_value=float(f["min"]), max_value=float(f["max"]),
                                                 value=None, help=f["help"] or None, key=k)
        else:
            inputs[f["name"]] = col.selectbox(label, f["options"], index=None, help=f["help"] or None, key=k)
    return inputs


def page_assess():
    st.header("Risk assessment")
    patients = api().patients()
    if not patients:
        st.info("Add a patient first.")
        return
    pmap = {f"{p['name']} ({p['mrn']})": p["id"] for p in patients}
    pid = pmap[st.selectbox("Patient", list(pmap))]
    models = api().models()
    if not models:
        st.error("No trained models found. Run: python -m prognosis.ml.train")
        return
    tabs = st.tabs([m["title"] for m in models])
    for tab, m in zip(tabs, models):
        with tab:
            spec = api().model(m["key"])
            st.caption(f"{m['description']}  Model: {m['selected']}, held-out AUC {m['roc_auc']:.3f}.")
            with st.form(f"form_{m['key']}"):
                inputs = feature_form(spec, m["key"])
                use_llm = st.checkbox("Rewrite the summary with an external language model (sends values only, never name or MRN)", key=f"llm_{m['key']}")
                go_ = st.form_submit_button("Estimate risk", type="primary")
            if go_:
                try:
                    pred = api().predict(pid, {"condition": m["key"], "inputs": inputs, "use_llm": use_llm})
                    st.session_state[f"last_{m['key']}"] = pred
                except ApiError as e:
                    show_errors(e)
            pred = st.session_state.get(f"last_{m['key']}")
            if pred and pred["patient_id"] == pid:
                res = pred["result"]
                c1, c2 = st.columns([1, 2])
                with c1:
                    st.plotly_chart(gauge(pred["probability"], res["thresholds"]), width="stretch")
                    st.markdown(f"**<span style='color:{BAND_COLOR[pred['band']]}'>{pred['band'].upper()} risk band</span>**", unsafe_allow_html=True)
                with c2:
                    st.subheader("What moved the estimate")
                    st.caption(f"Compared with a typical training patient (average {res['baseline']:.1%}).")
                    st.plotly_chart(contribution_chart(res), width="stretch")
                    n_empty = sum(1 for c in res["contributions"] if c["imputed"])
                    if n_empty:
                        st.caption(f"{n_empty} input(s) were left empty and are filled with typical training values, so they are not shown here.")
                for w in res["warnings"]:
                    st.warning(w)
                st.info(res["summary"])
                st.download_button("Download PDF report", api().report(pred["id"]), f"{m['key']}_report_{pred['id']}.pdf", "application/pdf", key=f"dl_{m['key']}")


def page_patients():
    st.header("Patients")
    with st.form("add_patient", clear_on_submit=True):
        c = st.columns(4)
        mrn = c[0].text_input("MRN")
        name = c[1].text_input("Name")
        sex = c[2].selectbox("Sex", ["Female", "Male"])
        by = c[3].number_input("Birth year", 1900, 2100, value=None)
        if st.form_submit_button("Add patient"):
            try:
                api().add_patient({"mrn": mrn, "name": name, "sex": sex, "birth_year": by})
                st.success("Patient added")
            except ApiError as e:
                show_errors(e)
    q = st.text_input("Search by name or MRN")
    rows = api().patients(q)
    if rows:
        st.dataframe(pd.DataFrame(rows).drop(columns=["created_at"]), width="stretch", hide_index=True)
    else:
        st.info("No patients found.")


def page_dashboard():
    st.header("Clinic overview")
    d = api().dashboard()
    c = st.columns(2)
    c[0].metric("Patients", d["patients"])
    c[1].metric("Patients with an assessment", d["assessments"])
    if not d["items"]:
        st.info("No assessments yet.")
        return
    fig = go.Figure()
    conds = list(d["counts"])
    for band in ("low", "moderate", "high"):
        fig.add_bar(name=band, x=conds, y=[d["counts"][k][band] for k in conds], marker_color=BAND_COLOR[band])
    fig.update_layout(barmode="stack", height=300, margin=dict(t=10), yaxis_title="patients (latest assessment)")
    st.plotly_chart(fig, width="stretch")
    df = pd.DataFrame(d["items"])
    f1, f2, f3 = st.columns(3)
    cond = f1.selectbox("Condition", ["All"] + sorted(df.condition.unique()))
    band = f2.selectbox("Band", ["All", "high", "moderate", "low"])
    s = f3.text_input("Search patient")
    if cond != "All": df = df[df.condition == cond]
    if band != "All": df = df[df.band == band]
    if s: df = df[df.patient.str.contains(s, case=False) | df.mrn.str.contains(s, case=False)]
    df = df.assign(risk=(df.probability * 100).round(1), when=pd.to_datetime(df["at"]).dt.strftime("%Y-%m-%d %H:%M"))
    show = df[["patient", "mrn", "condition", "risk", "band", "when"]].rename(columns={"risk": "risk %"})
    shade = {"high": "background-color:#ffcdd2", "moderate": "background-color:#ffe0b2", "low": "background-color:#c8e6c9"}
    st.dataframe(show.style.map(lambda v: shade.get(v, ""), subset=["band"]), width="stretch", hide_index=True)


def page_trends():
    st.header("Trends")
    patients = api().patients()
    if not patients:
        st.info("No patients yet.")
        return
    pmap = {f"{p['name']} ({p['mrn']})": p["id"] for p in patients}
    c = st.columns(2)
    pid = pmap[c[0].selectbox("Patient", list(pmap))]
    cond = c[1].selectbox("Condition", [m["key"] for m in api().models()])
    t = api().trend(pid, cond)
    if not t["points"]:
        st.info("No assessments for this patient and condition yet.")
        return
    df = pd.DataFrame(t["points"])
    df["at"] = pd.to_datetime(df["at"])
    th = api().model(cond)["card"]["thresholds"]
    fig = go.Figure(go.Scatter(x=df["at"], y=df["probability"] * 100, mode="lines+markers",
                               marker=dict(color=[BAND_COLOR[b] for b in df["band"]], size=11), line=dict(color="#555")))
    fig.add_hline(y=th["low_moderate"] * 100, line_dash="dot", line_color=BAND_COLOR["moderate"])
    fig.add_hline(y=th["moderate_high"] * 100, line_dash="dot", line_color=BAND_COLOR["high"])
    fig.update_layout(height=340, yaxis_title="risk %", margin=dict(t=10))
    st.plotly_chart(fig, width="stretch")
    s = t["summary"]
    if s["delta"] is None:
        st.info("Only one assessment so far, a trend needs at least two.")
    else:
        msg = f"{s['direction'].capitalize()}: {s['delta'] * 100:+.1f} points since the first assessment ({s['n']} assessments)."
        if s["band_change"]:
            msg += f" Band moved {s['band_change']}."
        (st.warning if s["direction"] == "rising" else st.success if s["direction"] == "falling" else st.info)(msg)
        st.caption("Each point is a fresh estimate from the inputs entered that day, so small moves can just be measurement noise.")


def page_models():
    st.header("Model cards")
    models = api().models()
    if not models:
        st.info("No trained models.")
        return
    key = st.selectbox("Model", [m["key"] for m in models], format_func=lambda k: next(m["title"] for m in models if m["key"] == k))
    card = api().model(key)["card"]
    t = card["test"]["metrics"]; ci = card["test"]["ci95"]
    st.caption(f"{card['label']}  Trained {card['trained_at'][:10]}  |  {card['data']['rows']} rows, prevalence {card['data']['prevalence']:.1%}")
    c = st.columns(4)
    c[0].metric("Selected", card["selected"])
    c[1].metric("Test ROC AUC", f"{t['roc_auc']:.3f}", f"CI {ci['roc_auc'][0]:.3f} to {ci['roc_auc'][1]:.3f}", delta_color="off")
    c[2].metric("Test Brier", f"{t['brier']:.3f}")
    c[3].metric("Calibration", card["calibration"])
    st.subheader("Cross-validated comparison (training split only)")
    tab = pd.DataFrame(card["cv"]["table"])[["model", "roc_auc", "roc_auc_sd", "brier_raw", "brier_cal", "ece_raw", "ece_cal", "calibration", "seconds"]]
    st.dataframe(tab.style.format({c: "{:.4f}" for c in tab.columns if c not in ("model", "calibration", "seconds")}), width="stretch", hide_index=True)
    st.subheader("Risk bands on the untouched test set")
    b = card["test"]["bands"]
    st.dataframe(pd.DataFrame([{"band": k, "patients": v["n"], "share": f"{v['share']:.0%}", "observed rate": f"{v['observed_rate']:.1%}" if v["observed_rate"] is not None else "n/a"} for k, v in b.items()]), hide_index=True)
    imp = card["importance"][:10][::-1]
    st.subheader("Permutation importance (test AUC drop)")
    st.plotly_chart(go.Figure(go.Bar(x=[i["auc_drop"] for i in imp], y=[i["feature"] for i in imp], orientation="h")).update_layout(height=320, margin=dict(t=10)), width="stretch")
    for k, v in card.get("extra", {}).items():
        st.caption(f"{k.replace('_', ' ')}: {v}")


def page_admin():
    st.header("Admin")
    with st.form("new_user", clear_on_submit=True):
        c = st.columns(3)
        u = c[0].text_input("Username"); p = c[1].text_input("Password (10+ chars)", type="password"); r = c[2].selectbox("Role", ["doctor", "admin"])
        if st.form_submit_button("Create user"):
            try:
                api().add_user({"username": u, "password": p, "role": r}); st.success("User created")
            except ApiError as e:
                show_errors(e)
    st.subheader("Audit log")
    st.dataframe(pd.DataFrame(api().audit()), width="stretch", hide_index=True)


def main():
    if "me" not in st.session_state:
        login_screen()
        return
    me = st.session_state.me
    st.sidebar.markdown(f"**{me['username']}**  \n{me['role']} at {me['clinic_name']}")
    pages = {"Risk assessment": page_assess, "Patients": page_patients, "Clinic overview": page_dashboard,
             "Trends": page_trends, "Model cards": page_models}
    if me["role"] == "admin":
        pages["Admin"] = page_admin
    choice = st.sidebar.radio("Go to", list(pages))
    if st.sidebar.button("Sign out"):
        st.session_state.clear()
        st.rerun()
    try:
        pages[choice]()
    except ApiError as e:
        if e.status == 401:
            st.session_state.clear()
            st.rerun()
        show_errors(e)


main()

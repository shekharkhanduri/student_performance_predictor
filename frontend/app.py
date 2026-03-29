"""
Faculty Student Diagnostic System — Streamlit Dashboard
========================================================
Pages
-----
1. 🏠 Global Pulse     – Class health KPIs, score histogram
2. 🔍 Student Discovery – Searchable / filterable student table
3. 📋 Diagnostic Report – Per-student SHAP waterfall + intervention simulator
4. ✏️  Manual Entry      – Single-student prediction form

Set the API_BASE_URL environment variable (default: http://localhost:8000).
"""

import os
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# ── Configuration ─────────────────────────────────────────────────────────────

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

PALETTE = {
    "navy": "#1B263B",
    "slate": "#415A77",
    "white": "#FFFFFF",
    "green": "#4CAF50",
    "orange": "#FF9800",
    "crimson": "#E63946",
    "bg": "#0D1B2A",
}

CATEGORICAL_OPTIONS: dict[str, list[str]] = {
    "Gender": ["Male", "Female"],
    "Parental_Involvement": ["Low", "Medium", "High"],
    "Access_to_Resources": ["Low", "Medium", "High"],
    "Extracurricular_Activities": ["Yes", "No"],
    "Motivation_Level": ["Low", "Medium", "High"],
    "Internet_Access": ["Yes", "No"],
    "Family_Income": ["Low", "Medium", "High"],
    "Teacher_Quality": ["Low", "Medium", "High"],
    "School_Type": ["Public", "Private"],
    "Peer_Influence": ["Positive", "Neutral", "Negative"],
    "Learning_Disabilities": ["No", "Yes"],
    "Parental_Education_Level": ["High School", "College", "Postgraduate"],
    "Distance_from_Home": ["Near", "Moderate", "Far"],
}

RISK_COLORS = {
    "Stable": PALETTE["green"],
    "Borderline": PALETTE["orange"],
    "At-Risk": PALETTE["crimson"],
}

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Faculty Diagnostic System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
        background-color: {PALETTE["bg"]};
        color: {PALETTE["white"]};
    }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background: {PALETTE["navy"]};
    }}

    /* Glass cards */
    .glass-card {{
        background: rgba(65, 90, 119, 0.35);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 16px;
        padding: 1.5rem 2rem;
        backdrop-filter: blur(10px);
        margin-bottom: 1rem;
    }}

    .metric-value {{
        font-size: 2.8rem;
        font-weight: 700;
        line-height: 1.1;
    }}
    .metric-label {{
        font-size: 0.85rem;
        color: rgba(255,255,255,0.65);
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }}

    /* Risk badges */
    .badge-stable   {{ background:{PALETTE["green"]};  color:#fff; padding:2px 10px; border-radius:999px; font-size:0.8rem; font-weight:600; }}
    .badge-border   {{ background:{PALETTE["orange"]}; color:#fff; padding:2px 10px; border-radius:999px; font-size:0.8rem; font-weight:600; }}
    .badge-atrisk   {{ background:{PALETTE["crimson"]}; color:#fff; padding:2px 10px; border-radius:999px; font-size:0.8rem; font-weight:600;
                      animation: pulse 1.5s infinite; }}

    @keyframes pulse {{
      0%   {{ box-shadow: 0 0 0 0 rgba(230,57,70,0.7); }}
      70%  {{ box-shadow: 0 0 0 8px rgba(230,57,70,0); }}
      100% {{ box-shadow: 0 0 0 0 rgba(230,57,70,0); }}
    }}

    h1, h2, h3 {{ color: {PALETTE["white"]}; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ── API helpers ───────────────────────────────────────────────────────────────


@st.cache_data(ttl=30)
def fetch_students(risk_level: str | None = None) -> list[dict]:
    params: dict = {"limit": 500}
    if risk_level:
        params["risk_level"] = risk_level
    try:
        r = requests.get(f"{API_BASE}/students", params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return []


def fetch_student(student_id: int) -> dict | None:
    try:
        r = requests.get(f"{API_BASE}/student/{student_id}", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def post_predict(payload: dict) -> dict | None:
    try:
        r = requests.post(f"{API_BASE}/predict", json=payload, timeout=60)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        return None


def upload_csv(file_bytes: bytes, filename: str) -> dict | None:
    try:
        r = requests.post(
            f"{API_BASE}/upload",
            files={"file": (filename, file_bytes, "text/csv")},
            timeout=300,
        )
        r.raise_for_status()
        return r.json()
    except requests.HTTPError as exc:
        detail = exc.response.json().get("detail", str(exc)) if exc.response else str(exc)
        st.error(f"Upload failed: {detail}")
        return None
    except Exception as exc:
        st.error(f"Upload failed: {exc}")
        return None


# ── Chart helpers ─────────────────────────────────────────────────────────────


def risk_badge_html(risk: str) -> str:
    css_map = {"Stable": "badge-stable", "Borderline": "badge-border", "At-Risk": "badge-atrisk"}
    cls = css_map.get(risk, "badge-stable")
    return f'<span class="{cls}">{risk}</span>'


def circular_gauge(value: float, label: str, color: str) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            title={"text": label, "font": {"color": "white", "size": 14}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "white"},
                "bar": {"color": color},
                "bgcolor": "rgba(0,0,0,0)",
                "bordercolor": "rgba(255,255,255,0.2)",
                "steps": [
                    {"range": [0, 60], "color": "rgba(230,57,70,0.15)"},
                    {"range": [60, 70], "color": "rgba(255,152,0,0.15)"},
                    {"range": [70, 100], "color": "rgba(76,175,80,0.15)"},
                ],
            },
            number={"font": {"color": "white", "size": 28}, "suffix": "%"},
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=40, b=20),
        height=200,
    )
    return fig


def score_histogram(students: list[dict], selected_bucket: str | None = None) -> go.Figure:
    scores = [s.get("predicted_exam_score", 0) for s in students]
    buckets = ["< 50", "50–59", "60–69", "70–79", "80–89", "≥ 90"]
    counts = [0] * 6
    for sc in scores:
        if sc < 50:
            counts[0] += 1
        elif sc < 60:
            counts[1] += 1
        elif sc < 70:
            counts[2] += 1
        elif sc < 80:
            counts[3] += 1
        elif sc < 90:
            counts[4] += 1
        else:
            counts[5] += 1

    bucket_colors = [
        PALETTE["crimson"],
        PALETTE["crimson"],
        PALETTE["orange"],
        PALETTE["orange"],
        PALETTE["green"],
        PALETTE["green"],
    ]
    bar_colors = [
        ("rgba(255,255,255,0.9)" if (selected_bucket and b == selected_bucket) else c)
        for b, c in zip(buckets, bucket_colors)
    ]

    fig = go.Figure(
        go.Bar(
            x=buckets,
            y=counts,
            marker_color=bar_colors,
            text=counts,
            textposition="outside",
            textfont={"color": "white"},
            hovertemplate="%{x}: %{y} students<extra></extra>",
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"},
        xaxis={"gridcolor": "rgba(255,255,255,0.1)"},
        yaxis={"gridcolor": "rgba(255,255,255,0.1)"},
        margin=dict(l=10, r=10, t=30, b=10),
        height=300,
        title={"text": "Student Distribution by Score Bucket", "font": {"color": "white"}},
    )
    return fig


def shap_waterfall(factors: list[dict]) -> go.Figure:
    features = [f["feature"] for f in factors]
    values = [f["shap_value"] for f in factors]
    colors = [PALETTE["crimson"] if v < 0 else PALETTE["green"] for v in values]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=features,
            orientation="h",
            marker_color=colors,
            text=[f"{v:+.2f}" for v in values],
            textposition="outside",
            textfont={"color": "white", "size": 12},
            hovertemplate="%{y}: %{x:+.3f} impact on score<extra></extra>",
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"},
        xaxis={
            "title": "SHAP Value (impact on predicted score)",
            "gridcolor": "rgba(255,255,255,0.1)",
            "zerolinecolor": "rgba(255,255,255,0.4)",
        },
        yaxis={"gridcolor": "rgba(255,255,255,0.1)"},
        margin=dict(l=160, r=60, t=40, b=40),
        height=max(250, len(factors) * 70),
        title={"text": "SHAP Impact — Top Risk Factors", "font": {"color": "white"}},
    )
    return fig


# ── Pages ──────────────────────────────────────────────────────────────────────


def page_global_pulse():
    st.title("🏠 Global Pulse")
    st.caption("Class-wide risk overview")

    students = fetch_students()

    if not students:
        st.info(
            "No student data yet. Upload a CSV via the sidebar or the **Upload** option, "
            "or use **Manual Entry** to add students."
        )
        return

    scores = [s.get("predicted_exam_score", 0) for s in students]
    avg_score = sum(scores) / len(scores) if scores else 0
    at_risk_count = sum(1 for s in students if s.get("risk_level") == "At-Risk")
    r2 = 98.82  # from training

    # ── Hero stat cards ────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    with col1:
        st.plotly_chart(
            circular_gauge(avg_score, "Class Health Index", PALETTE["green"]),
            use_container_width=True,
        )
    with col2:
        color = PALETTE["crimson"] if at_risk_count > 0 else PALETTE["green"]
        st.markdown(
            f"""<div class="glass-card" style="text-align:center">
                <div class="metric-value" style="color:{color}">{at_risk_count}</div>
                <div class="metric-label">⚠️ Intervention Alerts (At-Risk)</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""<div class="glass-card" style="text-align:center">
                <div class="metric-value" style="color:{PALETTE["green"]}">{r2}%</div>
                <div class="metric-label">🎯 Model R² Prediction Confidence</div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── At-Glance Histogram ────────────────────────────────────────────────
    st.subheader("At-Glance Score Distribution")
    bucket_options = ["All", "< 50", "50–59", "60–69", "70–79", "80–89", "≥ 90"]
    selected_bucket = st.selectbox("Filter by score bucket:", bucket_options, index=0)
    sb = None if selected_bucket == "All" else selected_bucket
    st.plotly_chart(score_histogram(students, sb), use_container_width=True)

    # Filtered student mini-list
    if sb:
        bucket_map = {
            "< 50": lambda s: s < 50,
            "50–59": lambda s: 50 <= s < 60,
            "60–69": lambda s: 60 <= s < 70,
            "70–79": lambda s: 70 <= s < 80,
            "80–89": lambda s: 80 <= s < 90,
            "≥ 90": lambda s: s >= 90,
        }
        filtered = [
            s for s in students
            if bucket_map[sb](s.get("predicted_exam_score", 0))
        ]
        if filtered:
            st.markdown(f"**{len(filtered)} student(s) in bucket {sb}:**")
            _render_student_table(filtered)


def _render_student_table(students: list[dict]):
    rows = []
    for s in students:
        rows.append(
            {
                "ID": s.get("student_id", "—"),
                "Name": s.get("student_name") or "—",
                "Score": round(s.get("predicted_exam_score", 0), 1),
                "Risk": s.get("risk_level", "—"),
                "Top Risk Factor": (
                    s["top_negative_factors"][0]["feature"]
                    if s.get("top_negative_factors")
                    else "—"
                ),
            }
        )
    df = pd.DataFrame(rows)

    def color_risk(val: str) -> str:
        return {
            "Stable": "color: #4CAF50",
            "Borderline": "color: #FF9800",
            "At-Risk": "color: #E63946",
        }.get(val, "")

    styled = df.style.map(color_risk, subset=["Risk"])
    st.dataframe(styled, use_container_width=True, hide_index=True)


def page_student_discovery():
    st.title("🔍 Student Discovery")
    st.caption("Search, filter, and navigate to individual diagnostic reports")

    # ── Filters ────────────────────────────────────────────────────────────
    with st.expander("🔧 Filters", expanded=True):
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            min_att = st.slider("Min Attendance (%)", 0, 100, 0)
            max_att = st.slider("Max Attendance (%)", 0, 100, 100)
        with fc2:
            involvement_filter = st.multiselect(
                "Parental Involvement",
                ["Low", "Medium", "High"],
                default=[],
            )
            risk_filter = st.multiselect(
                "Risk Level",
                ["Stable", "Borderline", "At-Risk"],
                default=[],
            )
        with fc3:
            min_hours = st.slider("Min Study Hours", 0, 44, 0)
            max_hours = st.slider("Max Study Hours", 0, 44, 44)

    search = st.text_input("🔍 Search by student name or ID:")

    students = fetch_students()
    if not students:
        st.info("No student data available. Upload a CSV first.")
        return

    # Apply filters
    filtered = students
    if min_att > 0 or max_att < 100:
        filtered = [
            s for s in filtered
            if min_att
            <= (s.get("features") or {}).get("Attendance", 100)
            <= max_att
        ]
    if involvement_filter:
        filtered = [
            s for s in filtered
            if (s.get("features") or {}).get("Parental_Involvement") in involvement_filter
        ]
    if risk_filter:
        filtered = [s for s in filtered if s.get("risk_level") in risk_filter]
    if min_hours > 0 or max_hours < 44:
        filtered = [
            s for s in filtered
            if min_hours
            <= (s.get("features") or {}).get("Hours_Studied", 44)
            <= max_hours
        ]
    if search:
        sl = search.lower()
        filtered = [
            s for s in filtered
            if sl in str(s.get("student_id", "")).lower()
            or sl in (s.get("student_name") or "").lower()
        ]

    st.markdown(f"**{len(filtered)} student(s) found**")

    if not filtered:
        return

    # Table with "View Diagnostic" button per row
    for s in filtered:
        risk = s.get("risk_level", "Unknown")
        score = s.get("predicted_exam_score", 0)
        name = s.get("student_name") or "—"
        sid = s.get("student_id")

        badge = risk_badge_html(risk)
        col1, col2, col3, col4, col5 = st.columns([1, 3, 2, 2, 2])
        with col1:
            st.markdown(f"**#{sid}**")
        with col2:
            st.markdown(name)
        with col3:
            st.markdown(f"**{round(score, 1)}**")
        with col4:
            st.markdown(badge, unsafe_allow_html=True)
        with col5:
            if st.button("📋 View Diagnostic", key=f"diag_{sid}"):
                st.session_state["selected_student_id"] = sid
                st.session_state["page"] = "📋 Diagnostic Report"
                st.rerun()

        st.markdown("<hr style='margin:4px 0; opacity:0.15'>", unsafe_allow_html=True)


def page_diagnostic_report():
    st.title("📋 Individual Diagnostic Report")

    sid = st.session_state.get("selected_student_id")
    manual_id = st.number_input("Enter Student ID:", min_value=1, value=sid or 1, step=1)
    if manual_id:
        sid = int(manual_id)

    if not sid:
        st.info("Select a student from the Discovery table or enter an ID above.")
        return

    student = fetch_student(sid)
    if not student:
        st.error(f"Student #{sid} not found.")
        return

    risk = student.get("risk_level", "Unknown")
    score = student.get("predicted_exam_score", 0)
    name = student.get("student_name") or f"Student #{sid}"
    factors = student.get("top_negative_factors") or []
    features = student.get("features") or {}

    # ── Bio header ──────────────────────────────────────────────────────────
    badge = risk_badge_html(risk)
    st.markdown(
        f"""<div class="glass-card">
            <h2 style="margin:0">{name} &nbsp; {badge}</h2>
            <p style="color:rgba(255,255,255,0.6); margin:4px 0">Student ID: {sid} &nbsp;|&nbsp;
            Predicted Score: <strong style="font-size:1.4rem">{round(score, 1)}</strong>/100</p>
        </div>""",
        unsafe_allow_html=True,
    )

    col_left, col_right = st.columns([1, 1])

    with col_left:
        # ── Score gauge ─────────────────────────────────────────────────────
        gauge_color = RISK_COLORS.get(risk, PALETTE["green"])
        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number+delta",
                value=score,
                delta={"reference": 70, "valueformat": ".1f"},
                title={"text": "Predicted Exam Score", "font": {"color": "white"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "white"},
                    "bar": {"color": gauge_color},
                    "bgcolor": "rgba(0,0,0,0)",
                    "steps": [
                        {"range": [0, 60], "color": "rgba(230,57,70,0.15)"},
                        {"range": [60, 70], "color": "rgba(255,152,0,0.15)"},
                        {"range": [70, 100], "color": "rgba(76,175,80,0.15)"},
                    ],
                    "threshold": {
                        "line": {"color": "white", "width": 3},
                        "thickness": 0.8,
                        "value": 70,
                    },
                },
                number={"font": {"color": "white", "size": 40}, "suffix": ""},
            )
        )
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font={"color": "white"},
            height=280,
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_right:
        # ── Advice generator ────────────────────────────────────────────────
        if factors:
            st.markdown("### 💡 Recommended Actions")
            for f in factors:
                fname = f.get("feature", "Unknown")
                desc = f.get("description", "")
                sv = f.get("shap_value", 0)
                st.markdown(
                    f"""<div class="glass-card" style="padding:1rem">
                        <strong>{fname}</strong>
                        <span style="color:rgba(230,57,70,0.9); margin-left:8px">
                        ({sv:+.2f} impact)</span>
                        {"<br><small>" + desc + "</small>" if desc else ""}
                    </div>""",
                    unsafe_allow_html=True,
                )

    st.markdown("---")

    # ── SHAP Waterfall ──────────────────────────────────────────────────────
    if factors:
        st.subheader("📊 SHAP Impact Waterfall")
        st.plotly_chart(shap_waterfall(factors), use_container_width=True)
    else:
        st.info("No SHAP explanations available for this student.")

    st.markdown("---")

    # ── Intervention Slider (LIME Simulator) ───────────────────────────────
    st.subheader("🎛️ Intervention Simulator (What-If Analysis)")
    st.caption("Adjust feature values to see the estimated score change.")

    with st.form("simulator_form"):
        sim_features = dict(features)

        c1, c2 = st.columns(2)
        with c1:
            sim_features["Hours_Studied"] = st.slider(
                "Hours Studied",
                0.0, 44.0,
                float(features.get("Hours_Studied", 5)),
                0.5,
            )
            sim_features["Attendance"] = st.slider(
                "Attendance (%)",
                60.0, 100.0,
                float(features.get("Attendance", 80)),
                1.0,
            )
            sim_features["Sleep_Hours"] = st.slider(
                "Sleep Hours",
                4.0, 10.0,
                float(features.get("Sleep_Hours", 7)),
                0.5,
            )
            sim_features["Tutoring_Sessions"] = st.slider(
                "Tutoring Sessions",
                0, 8,
                int(features.get("Tutoring_Sessions", 0)),
            )
        with c2:
            sim_features["Motivation_Level"] = st.selectbox(
                "Motivation Level",
                CATEGORICAL_OPTIONS["Motivation_Level"],
                index=CATEGORICAL_OPTIONS["Motivation_Level"].index(
                    features.get("Motivation_Level", "Medium")
                )
                if features.get("Motivation_Level") in CATEGORICAL_OPTIONS["Motivation_Level"]
                else 1,
            )
            sim_features["Parental_Involvement"] = st.selectbox(
                "Parental Involvement",
                CATEGORICAL_OPTIONS["Parental_Involvement"],
                index=CATEGORICAL_OPTIONS["Parental_Involvement"].index(
                    features.get("Parental_Involvement", "Medium")
                )
                if features.get("Parental_Involvement") in CATEGORICAL_OPTIONS["Parental_Involvement"]
                else 1,
            )
            sim_features["Internet_Access"] = st.selectbox(
                "Internet Access",
                CATEGORICAL_OPTIONS["Internet_Access"],
                index=CATEGORICAL_OPTIONS["Internet_Access"].index(
                    features.get("Internet_Access", "Yes")
                )
                if features.get("Internet_Access") in CATEGORICAL_OPTIONS["Internet_Access"]
                else 0,
            )
            sim_features["Access_to_Resources"] = st.selectbox(
                "Access to Resources",
                CATEGORICAL_OPTIONS["Access_to_Resources"],
                index=CATEGORICAL_OPTIONS["Access_to_Resources"].index(
                    features.get("Access_to_Resources", "Medium")
                )
                if features.get("Access_to_Resources") in CATEGORICAL_OPTIONS["Access_to_Resources"]
                else 1,
            )

        submitted = st.form_submit_button("🔮 Simulate Score")

    if submitted:
        with st.spinner("Computing simulated prediction…"):
            sim_result = post_predict({**sim_features, "student_name": None})
        if sim_result:
            sim_score = sim_result.get("predicted_exam_score", 0)
            delta = sim_score - score
            delta_color = PALETTE["green"] if delta >= 0 else PALETTE["crimson"]
            arrow = "▲" if delta >= 0 else "▼"
            st.markdown(
                f"""<div class="glass-card" style="text-align:center">
                    <div class="metric-value">{round(sim_score, 1)}</div>
                    <div style="color:{delta_color}; font-size:1.2rem; font-weight:600">
                    {arrow} {abs(delta):.1f} pts vs. current</div>
                    <div class="metric-label">Simulated Score — {sim_result.get("risk_level")}</div>
                </div>""",
                unsafe_allow_html=True,
            )


def page_manual_entry():
    st.title("✏️ Manual Entry")
    st.caption("Enter feature values for a single student to get an instant prediction.")

    with st.form("manual_entry_form"):
        st.markdown("#### 📊 Numeric Features")
        c1, c2, c3 = st.columns(3)
        with c1:
            hours = st.number_input("Hours Studied", 0.0, 44.0, 8.0, 0.5)
            attendance = st.number_input("Attendance (%)", 0.0, 100.0, 80.0, 1.0)
            sleep = st.number_input("Sleep Hours", 0.0, 12.0, 7.0, 0.5)
        with c2:
            prev_scores = st.number_input("Previous Scores", 0.0, 100.0, 70.0, 1.0)
            tutoring = st.number_input("Tutoring Sessions", 0, 8, 0)
            physical = st.number_input("Physical Activity (hrs/week)", 0, 6, 3)
        with c3:
            student_name = st.text_input("Student Name (optional)")

        st.markdown("#### 🏷️ Categorical Features")
        ca1, ca2, ca3 = st.columns(3)
        with ca1:
            gender = st.selectbox("Gender", CATEGORICAL_OPTIONS["Gender"])
            parental_inv = st.selectbox("Parental Involvement", CATEGORICAL_OPTIONS["Parental_Involvement"])
            access_res = st.selectbox("Access to Resources", CATEGORICAL_OPTIONS["Access_to_Resources"])
            extra = st.selectbox("Extracurricular Activities", CATEGORICAL_OPTIONS["Extracurricular_Activities"])
            motivation = st.selectbox("Motivation Level", CATEGORICAL_OPTIONS["Motivation_Level"])
        with ca2:
            internet = st.selectbox("Internet Access", CATEGORICAL_OPTIONS["Internet_Access"])
            family_inc = st.selectbox("Family Income", CATEGORICAL_OPTIONS["Family_Income"])
            teacher_q = st.selectbox("Teacher Quality", CATEGORICAL_OPTIONS["Teacher_Quality"])
            school = st.selectbox("School Type", CATEGORICAL_OPTIONS["School_Type"])
        with ca3:
            peer = st.selectbox("Peer Influence", CATEGORICAL_OPTIONS["Peer_Influence"])
            disabilities = st.selectbox("Learning Disabilities", CATEGORICAL_OPTIONS["Learning_Disabilities"])
            parental_edu = st.selectbox("Parental Education Level", CATEGORICAL_OPTIONS["Parental_Education_Level"])
            distance = st.selectbox("Distance from Home", CATEGORICAL_OPTIONS["Distance_from_Home"])

        submitted = st.form_submit_button("🔮 Predict & Save")

    if submitted:
        payload: dict[str, Any] = {
            "student_name": student_name or None,
            "Hours_Studied": hours,
            "Attendance": attendance,
            "Gender": gender,
            "Parental_Involvement": parental_inv,
            "Access_to_Resources": access_res,
            "Extracurricular_Activities": extra,
            "Sleep_Hours": sleep,
            "Previous_Scores": prev_scores,
            "Motivation_Level": motivation,
            "Internet_Access": internet,
            "Tutoring_Sessions": float(tutoring),
            "Family_Income": family_inc,
            "Teacher_Quality": teacher_q,
            "School_Type": school,
            "Peer_Influence": peer,
            "Physical_Activity": float(physical),
            "Learning_Disabilities": disabilities,
            "Parental_Education_Level": parental_edu,
            "Distance_from_Home": distance,
        }
        with st.spinner("Running prediction…"):
            result = post_predict(payload)

        if result:
            risk = result.get("risk_level", "Unknown")
            score = result.get("predicted_exam_score", 0)
            badge = risk_badge_html(risk)
            factors = result.get("top_negative_factors") or []

            st.markdown(
                f"""<div class="glass-card" style="text-align:center">
                    <div class="metric-value">{round(score, 1)}</div>
                    <div style="margin-top:6px">{badge}</div>
                    <div class="metric-label" style="margin-top:8px">
                    Predicted Exam Score · Student saved as ID #{result.get("student_id")}</div>
                </div>""",
                unsafe_allow_html=True,
            )

            if factors:
                st.markdown("### 📊 SHAP Waterfall")
                st.plotly_chart(shap_waterfall(factors), use_container_width=True)

                st.markdown("### 💡 Recommended Actions")
                for f in factors:
                    desc = f.get("description", "")
                    fname = f.get("feature", "")
                    sv = f.get("shap_value", 0)
                    st.markdown(
                        f"- **{fname}** ({sv:+.2f}): {desc}"
                    )


# ── Upload helper page ─────────────────────────────────────────────────────────


def sidebar_upload():
    st.sidebar.markdown("---")
    st.sidebar.subheader("📤 Upload CSV")
    uploaded = st.sidebar.file_uploader(
        "Upload student CSV", type=["csv"], label_visibility="collapsed"
    )
    if uploaded:
        if st.sidebar.button("Process Upload"):
            with st.spinner("Uploading and running predictions…"):
                result = upload_csv(uploaded.read(), uploaded.name)
            if result:
                st.sidebar.success(
                    f"✅ {result['rows_stored']}/{result['rows_processed']} rows stored "
                    f"(batch: {result['batch_id']})"
                )
                if result.get("errors"):
                    st.sidebar.warning(f"⚠️ {len(result['errors'])} row errors.")
                fetch_students.clear()


# ── Navigation ────────────────────────────────────────────────────────────────

PAGE_MAP = {
    "🏠 Global Pulse": page_global_pulse,
    "🔍 Student Discovery": page_student_discovery,
    "📋 Diagnostic Report": page_diagnostic_report,
    "✏️ Manual Entry": page_manual_entry,
}

st.sidebar.title("🎓 Faculty Diagnostic System")
st.sidebar.caption("Stacking Ensemble · XGBoost + CatBoost + RF + LassoCV")

if "page" not in st.session_state:
    st.session_state["page"] = "🏠 Global Pulse"

page = st.sidebar.radio(
    "Navigation",
    list(PAGE_MAP.keys()),
    index=list(PAGE_MAP.keys()).index(st.session_state["page"]),
)
st.session_state["page"] = page

sidebar_upload()

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"**API:** `{API_BASE}`  \n"
    "Set `API_BASE_URL` env var to change."
)

# ── Render selected page ───────────────────────────────────────────────────────
PAGE_MAP[page]()

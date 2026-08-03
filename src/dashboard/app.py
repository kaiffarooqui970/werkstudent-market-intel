"""
Werkstudent Market Intelligence — Streamlit dashboard.

Tabs:
  1. Overview       — headline numbers
  2. Skills         — demand by skill × city × time
  3. Language gate  — English-friendly filter
  4. Werkstudent pay — pay bands
  5. Salary model    — XGBoost predictor + SHAP

Run: streamlit run src/dashboard/app.py
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

import duckdb
import pandas as pd
import plotly.express as px
import streamlit as st

DB_PATH = os.environ.get("WSI_DB", "data/warehouse.duckdb")
MODEL_PATH = os.environ.get("WSI_MODEL", "data/model.pkl")

st.set_page_config(
    page_title="Werkstudent Market Intel",
    page_icon="📊",
    layout="wide",
)


@st.cache_resource
def get_con() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(DB_PATH, read_only=True)


@st.cache_resource
def get_model():
    if not Path(MODEL_PATH).exists():
        return None
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_data(ttl=300)
def q(sql: str) -> pd.DataFrame:
    return get_con().execute(sql).fetch_df()


def _guarded():
    try:
        n = q("SELECT COUNT(*) AS n FROM fct_jobs")["n"].iloc[0]
    except Exception as e:
        st.error(f"warehouse not ready: {e}\n\nRun `make demo` or `make ingest && make transform`.")
        st.stop()
    if n == 0:
        st.warning("fct_jobs is empty. Run `make ingest && make transform` first.")
        st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────

st.title("📊 Werkstudent Market Intelligence")
st.caption(
    "Live analytics on the German student-job market. Data via Arbeitnow. "
    "Built as a portfolio project — see the [repo](https://github.com/) for methodology."
)

_guarded()

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar filters
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.subheader("Filters")
    cities = q("SELECT DISTINCT city FROM fct_jobs ORDER BY 1")["city"].tolist()
    role_families = q("SELECT DISTINCT role_family FROM fct_jobs ORDER BY 1")["role_family"].tolist()

    sel_cities = st.multiselect("City", cities, default=cities)
    sel_roles = st.multiselect("Role family", role_families, default=role_families)
    english_only = st.checkbox("English-friendly only", value=False)

    st.markdown("---")
    st.caption(f"DB: `{DB_PATH}`")


city_filter = "', '".join(sel_cities) if sel_cities else ""
role_filter = "', '".join(sel_roles) if sel_roles else ""
where_clauses = [f"city IN ('{city_filter}')" if sel_cities else "TRUE",
                 f"role_family IN ('{role_filter}')" if sel_roles else "TRUE"]
if english_only:
    where_clauses.append("language_gate = 'english_ok'")
WHERE = " AND ".join(where_clauses)

tabs = st.tabs(["Overview", "Skills", "Language gate", "Werkstudent pay", "Salary model"])

# ─────────────────────────────────────────────────────────────────────────────
# Overview
# ─────────────────────────────────────────────────────────────────────────────

with tabs[0]:
    kpi = q(f"""
        SELECT
            COUNT(*)                                                     AS n_jobs,
            COUNT(DISTINCT company_name)                                  AS n_companies,
            SUM(CASE WHEN remote THEN 1 ELSE 0 END)                       AS n_remote,
            SUM(CASE WHEN language_gate = 'english_ok' THEN 1 ELSE 0 END) AS n_english,
            AVG(salary_min_annualized_eur)                                AS avg_salary
        FROM fct_jobs WHERE {WHERE}
    """).iloc[0]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Postings", f"{int(kpi['n_jobs']):,}")
    c2.metric("Companies", f"{int(kpi['n_companies']):,}")
    c3.metric("Remote", f"{int(kpi['n_remote']):,}")
    c4.metric("English-friendly", f"{int(kpi['n_english']):,}")
    avg = kpi["avg_salary"]
    c5.metric("Avg. salary (min)", f"€{int(avg):,}" if pd.notna(avg) else "—")

    st.markdown("### Postings by role family")
    df = q(f"""
        SELECT role_family, COUNT(*) AS n
        FROM fct_jobs WHERE {WHERE}
        GROUP BY 1 ORDER BY n DESC
    """)
    if not df.empty:
        st.plotly_chart(px.bar(df, x="role_family", y="n"), use_container_width=True)

    st.markdown("### Postings by city")
    df = q(f"""
        SELECT city, COUNT(*) AS n
        FROM fct_jobs WHERE {WHERE}
        GROUP BY 1 ORDER BY n DESC LIMIT 15
    """)
    if not df.empty:
        st.plotly_chart(px.bar(df, x="city", y="n"), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Skills
# ─────────────────────────────────────────────────────────────────────────────

with tabs[1]:
    st.markdown("### Most demanded skills")
    df = q(f"""
        SELECT b.skill_name, d.category, COUNT(*) AS n
        FROM fct_jobs f
        JOIN bridge_job_skill b USING (slug)
        LEFT JOIN dim_skills d ON d.skill_name = b.skill_name
        WHERE {WHERE}
        GROUP BY 1, 2
        ORDER BY n DESC LIMIT 25
    """)
    if df.empty:
        st.info("no skills extracted in current filter")
    else:
        st.plotly_chart(
            px.bar(df, x="n", y="skill_name", color="category", orientation="h",
                   height=600).update_yaxes(categoryorder="total ascending"),
            use_container_width=True,
        )

    st.markdown("### Skill × city heatmap")
    df = q(f"""
        SELECT b.skill_name, f.city, COUNT(*) AS n
        FROM fct_jobs f JOIN bridge_job_skill b USING (slug)
        WHERE {WHERE}
        GROUP BY 1, 2
    """)
    if not df.empty:
        top_skills = (df.groupby("skill_name")["n"].sum()
                        .sort_values(ascending=False).head(15).index.tolist())
        top_cities_ = (df.groupby("city")["n"].sum()
                          .sort_values(ascending=False).head(10).index.tolist())
        pivot = (df[df["skill_name"].isin(top_skills) & df["city"].isin(top_cities_)]
                 .pivot_table(index="skill_name", columns="city", values="n", fill_value=0))
        st.plotly_chart(
            px.imshow(pivot, aspect="auto", color_continuous_scale="Blues"),
            use_container_width=True,
        )

# ─────────────────────────────────────────────────────────────────────────────
# Language gate
# ─────────────────────────────────────────────────────────────────────────────

with tabs[2]:
    st.markdown("### German language requirement by city × role family")
    df = q(f"""
        SELECT city, role_family, pct_german_required, pct_english_ok, n_postings
        FROM mart_language_gate
        WHERE city IN ('{city_filter}') AND role_family IN ('{role_filter}')
    """) if sel_cities and sel_roles else pd.DataFrame()
    if df.empty:
        st.info("not enough postings for language-gate stats in current filter")
    else:
        st.dataframe(df.sort_values("pct_english_ok", ascending=False), use_container_width=True)

        pivot = df.pivot_table(
            index="city", columns="role_family", values="pct_german_required", fill_value=None
        )
        st.plotly_chart(
            px.imshow(pivot, aspect="auto", color_continuous_scale="Reds",
                      labels=dict(color="% German required")),
            use_container_width=True,
        )

    st.markdown("### Skills gated by German")
    df = q(f"""
        SELECT skill_name,
               SUM(n_postings) AS total,
               SUM(n_german_required) AS n_german,
               ROUND(100.0 * SUM(n_german_required) / NULLIF(SUM(n_postings), 0), 1) AS pct_german
        FROM mart_skill_demand
        GROUP BY 1
        HAVING total >= 5
        ORDER BY pct_german DESC LIMIT 20
    """)
    if not df.empty:
        st.dataframe(df, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Werkstudent pay
# ─────────────────────────────────────────────────────────────────────────────

with tabs[3]:
    st.markdown("### Pay bands (annualized) for Werkstudent / Intern / Junior")
    st.caption(
        "Hourly rates are annualized at 20h/week × 52 weeks for comparability. "
        "Cells shown only where ≥ 2 postings had a reported salary."
    )
    df = q("SELECT * FROM mart_werkstudent_pay ORDER BY median_min_eur DESC")
    if df.empty:
        st.info("no pay data yet (Arbeitnow salary coverage is sparse)")
    else:
        st.dataframe(df, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Salary model
# ─────────────────────────────────────────────────────────────────────────────

with tabs[4]:
    st.markdown("### Salary prediction (XGBoost + SHAP)")
    bundle = get_model()
    if bundle is None:
        st.warning(
            "No model trained yet. Run `make train` after ingest/transform. "
            "Training needs at least 5 salaried postings per model feature "
            "(skills + city + role + seniority + language dummies), 30 rows minimum — "
            "check the training log for the exact row count required."
        )
    else:
        cA, cB = st.columns([1, 1])
        with cA:
            st.markdown("#### Model quality (grouped CV)")
            m = bundle.metrics
            st.write({
                "MAE (log-EUR)": round(m.get("mae_log", float("nan")), 3),
                "R²":            round(m.get("r2", float("nan")), 3),
                "Folds":         m.get("n_folds", m.get("n_repeats")),
            })
            st.caption(
                "MAE is on log-transformed salary — 0.20 ≈ ±22% relative error. "
                "R² can be modest given how noisy self-reported salary ranges are."
            )
        with cB:
            st.markdown("#### Top drivers (global mean |SHAP|)")
            shap_df = pd.DataFrame(
                [{"feature": k, "importance": v} for k, v in bundle.shap_global.items()]
            )
            st.plotly_chart(
                px.bar(shap_df, x="importance", y="feature", orientation="h", height=520)
                  .update_yaxes(categoryorder="total ascending"),
                use_container_width=True,
            )

        st.markdown("#### Try it")
        c1, c2, c3 = st.columns(3)
        with c1:
            city_in = st.selectbox("City", bundle.top_cities + ["Other"], index=0)
            role_in = st.selectbox("Role family",
                                    ["DS/ML", "Analytics", "Data Eng", "AI", "Software Eng", "Other"])
            sen_in = st.selectbox("Seniority",
                                   ["werkstudent", "intern", "junior", "mid", "senior", "unknown"])
        with c2:
            remote_in = st.checkbox("Remote", value=False)
            lang_in = st.selectbox("Language gate", ["english_ok", "unclear", "german_required"])
        with c3:
            skills_in = st.multiselect("Skills", bundle.all_skills,
                                        default=[s for s in ["Python", "SQL"] if s in bundle.all_skills])

        from src.ml.salary_model import predict_eur
        pred = predict_eur(
            bundle,
            skills=skills_in, city=city_in, role_family=role_in,
            seniority=sen_in, remote=remote_in, language_gate=lang_in,
        )
        st.metric("Predicted min salary (annualized)", f"€{int(pred):,}")
        st.caption(
            "Point estimate only. Trained on the ~20% of postings that report salary — "
            "treat as directional, not authoritative."
        )

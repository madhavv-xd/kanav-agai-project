import json
from pathlib import Path
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"

st.set_page_config(page_title="AI Workforce Intelligence", layout="wide")
st.title("AI Workforce Intelligence Platform")
st.caption("Enterprise HR AI — Attrition, Engagement, Skill Gaps & Upskilling")

scores = pd.read_csv(DATA / "attrition_scores.csv")
metrics = pd.read_csv(DATA / "model_metrics.csv").iloc[0]
gaps = pd.read_csv(DATA / "organization_skill_gaps.csv")
intel = pd.read_csv(DATA / "employee_intelligence.csv")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Employees", f"{len(scores):,}")
c2.metric("High Risk", f"{(scores.Risk == 'HIGH').sum():,}")
c3.metric("Avg Attrition Probability", f"{scores.Attrition_Prob.mean():.1%}")
c4.metric("ROC-AUC", f"{metrics.roc_auc:.3f}")

st.divider()

st.subheader("Attrition Risk by Department")
dept = scores.groupby("Department")["Attrition_Prob"].mean().sort_values(ascending=False)
st.bar_chart(dept)

st.subheader("Risk Distribution")
risk_counts = scores["Risk"].value_counts().reindex(["LOW", "MEDIUM", "HIGH"]).fillna(0)
st.bar_chart(risk_counts)

st.subheader("Critical Organisation Skill Gaps")
if gaps.empty:
    st.info("No role-mapped skill gaps were produced. Add/verify O*NET role mappings.")
else:
    st.dataframe(gaps.head(20), use_container_width=True)

st.subheader("Upskilling Recommendations")
st.dataframe(
    intel[["EmployeeNumber", "JobRole", "Attrition_Prob", "Risk", "SkillGap", "Recommendation"]]
    .head(50),
    use_container_width=True
)

st.subheader("Employee Drill-down")
employee_id = st.number_input("EmployeeNumber", min_value=1, value=int(scores.EmployeeNumber.iloc[0]))
row = intel[intel.EmployeeNumber == employee_id]
if row.empty:
    st.warning("Employee not found in the MVP intelligence table.")
else:
    st.json(row.iloc[0].to_dict())

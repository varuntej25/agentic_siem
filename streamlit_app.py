import streamlit as st
import subprocess
import json
import os
import time
import glob
import pandas as pd

st.set_page_config(page_title="Spharaka Security Dashboard", layout="centered")

st.markdown("<h2 style='text-align: center; color: #2E86C1;'>🛡️ Spharaka Security Investigation Dashboard</h2>", unsafe_allow_html=True)

# Centered input boxes
col1, col2 = st.columns(2)
with col1:
    alert_id = st.text_input("Enter Alert ID(s):", "12,15")
with col2:
    command = st.text_input("Enter Command:", "investigate")

if st.button("🚀 Run Analysis", use_container_width=True):
    with st.spinner("⏳ Running analysis... This may take a few minutes, please wait."):
        # Run main_orchestrator.py (no args needed, it uses default config/paths)
        subprocess.run(["python3", "main_orchestrator.py"])

        # Wait until a JSON file appears in reports/
        timeout = 300  # 5 min
        report_path = None
        start = time.time()
        while time.time() - start < timeout:
            files = glob.glob("reports/*.json")
            if files:
                report_path = max(files, key=os.path.getctime)  # latest file
                break
            time.sleep(2)

        if report_path:
            with open(report_path, "r") as f:
                report = json.load(f)

            st.success("✅ Report Generated Successfully!")

            # ---- IP AGENT ----
            if "ip_agent" in report:
                st.subheader("🌍 IP Agent Analysis")
                try:
                    st.dataframe(pd.DataFrame(report["ip_agent"]))
                except:
                    st.json(report["ip_agent"])

            # ---- USER CONTEXT ----
            if "user_context" in report:
                st.subheader("👤 User Context")
                st.json(report["user_context"])

            # ---- ELASTIC SEARCH ----
            if "elastic_search" in report:
                st.subheader("📊 Elastic Search Queries")
                st.json(report["elastic_search"])

            # ---- ALERTS VIEWER ----
            if "alerts" in report:
                st.subheader("🚨 Alerts Viewer")
                try:
                    st.dataframe(pd.DataFrame(report["alerts"]))
                except:
                    st.json(report["alerts"])

            # ---- TIMELINE ----
            if "timeline" in report:
                st.subheader("⏳ Timeline Analysis")
                st.json(report["timeline"])  # later replace with Plotly timeline

            # ---- SUMMARY ----
            if "summary" in report:
                st.subheader("📑 Summary")
                st.json(report["summary"])

            # ---- Download JSON ----
            st.download_button(
                label="⬇️ Download Full Report",
                data=json.dumps(report, indent=2),
                file_name=os.path.basename(report_path),
                mime="application/json"
            )
        else:
            st.error("❌ Timeout: Report file not generated within 5 minutes.")

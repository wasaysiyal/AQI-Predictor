# src/hopsworks_client.py
import os
import threading
import hopsworks

_lock = threading.Lock()
_project = None

def get_hopsworks_project():
    global _project
    with _lock:
        if _project is not None:
            return _project

        api_key = os.environ.get("HOPSWORKS_API_KEY")
        project_name = os.environ.get("HOPSWORKS_PROJECT", "aqi_predictorrr")

        # Streamlit secrets (only available when running streamlit)
        try:
            import streamlit as st
            if not api_key and "HOPSWORKS_API_KEY" in st.secrets:
                api_key = st.secrets["HOPSWORKS_API_KEY"]
            if "HOPSWORKS_PROJECT" in st.secrets:
                project_name = st.secrets["HOPSWORKS_PROJECT"]
        except Exception:
            pass

        if not api_key:
            raise RuntimeError(
                "Missing HOPSWORKS_API_KEY. Set it as env var OR in .streamlit/secrets.toml"
            )

        _project = hopsworks.login(project=project_name, api_key_value=api_key)
        return _project

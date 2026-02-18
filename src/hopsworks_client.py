import os
import streamlit as st
import hopsworks


def _get_secret(name: str, default: str | None = None) -> str | None:
    # 1) env var
    v = os.getenv(name)
    if v:
        return v.strip()

    # 2) streamlit secrets
    try:
        if name in st.secrets:
            return str(st.secrets[name]).strip()
    except Exception:
        pass

    return default


def get_hopsworks_project():
    # ---- FIX: Windows temp directory for certificates ----
    os.makedirs("C:/temp", exist_ok=True)
    os.environ["TMP"] = "C:/temp"
    os.environ["TEMP"] = "C:/temp"

    api_key = _get_secret("HOPSWORKS_API_KEY")
    project_name = _get_secret("HOPSWORKS_PROJECT")

    if not api_key:
        raise RuntimeError("Missing HOPSWORKS_API_KEY.")
    if not project_name:
        raise RuntimeError("Missing HOPSWORKS_PROJECT.")

    return hopsworks.login(project=project_name, api_key_value=api_key)

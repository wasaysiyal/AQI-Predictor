# src/hopsworks_client.py
import os
import streamlit as st
import hopsworks


def _get_secret(name: str, default: str | None = None) -> str | None:
    # 1) env var (GitHub Actions / local)
    v = os.getenv(name)
    if v:
        return v.strip()

    # 2) streamlit secrets (local streamlit / deployed)
    try:
        if name in st.secrets:
            return str(st.secrets[name]).strip()
    except Exception:
        pass

    return default


def get_hopsworks_project():
    api_key = _get_secret("HOPSWORKS_API_KEY")
    project_name = _get_secret("HOPSWORKS_PROJECT")

    if not api_key:
        raise RuntimeError("Missing HOPSWORKS_API_KEY. Set env var or .streamlit/secrets.toml")
    if not project_name:
        raise RuntimeError("Missing HOPSWORKS_PROJECT. Set env var or .streamlit/secrets.toml")

    return hopsworks.login(project=project_name, api_key_value=api_key)

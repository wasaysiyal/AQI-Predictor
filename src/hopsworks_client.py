# src/hopsworks_client.py
import os
import hopsworks

def get_hopsworks_project():
    api_key = os.getenv("HOPSWORKS_API_KEY", "")
    project_name = os.getenv("HOPSWORKS_PROJECT", "")

    api_key = api_key.strip()
    project_name = project_name.strip()

    if not api_key:
        raise RuntimeError("Missing HOPSWORKS_API_KEY (env var).")
    if not project_name:
        raise RuntimeError("Missing HOPSWORKS_PROJECT (env var).")

    return hopsworks.login(project=project_name, api_key_value=api_key)

# in src/evaluate_predictions.py right before merge
import pandas as pd

pred["event_time"] = pd.to_datetime(pred["event_time"], utc=True, errors="coerce")
feat["event_time"] = pd.to_datetime(feat["event_time"], utc=True, errors="coerce")

df = pred.merge(feat, on="event_time", how="left").sort_values("event_time")


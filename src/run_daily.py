# src/run_daily.py
import subprocess
import sys

def run(cmd: list[str]):
    print("\n>>>", " ".join(cmd))
    subprocess.check_call(cmd)

def main():
    run([sys.executable, "-c",
         "from src.feature_store_upload import upload_daily_features; upload_daily_features(days=3)"])
    run([sys.executable, "-m", "src.training_dataset"])
    run([sys.executable, "-m", "src.train"])
    run([sys.executable, "-m", "src.batch_inference"])
    print("\n✅ Daily pipeline finished")

if __name__ == "__main__":
    main()

import subprocess
import sys

def run(cmd):
    print(f"\n>>> {cmd}")
    r = subprocess.run(cmd, shell=True)
    if r.returncode != 0:
        sys.exit(r.returncode)

if __name__ == "__main__":
    run("python -m src.feature_store_upload")
    run("python -m src.training_dataset")
    run("python -m src.train")
    run("python -m src.batch_inference")
    print("\n✅ Pipeline finished")

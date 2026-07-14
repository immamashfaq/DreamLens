import os
import sys
# Reconfigure stdout to support UTF-8 on Windows
sys.stdout.reconfigure(encoding='utf-8')
import argparse
import httpx
from tqdm import tqdm

BASE_URL = "https://physionet.org/files/hmc-sleep-staging/1.1/recordings/"
LOCAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset", "hmc-sleep-data")

def download_file(client, filename):
    url = f"{BASE_URL}{filename}"
    filepath = os.path.join(LOCAL_DIR, filename)
    
    if os.path.exists(filepath):
        print(f"   [INFO] {filename} already exists. Skipping.")
        return True
        
    print(f"   [DOWNLOAD] Downloading {filename}...")
    try:
        # Stream the download to avoid holding large files in memory
        with client.stream("GET", url) as response:
            if response.status_code != 200:
                print(f"   [ERROR] Failed to download {filename} (HTTP Status {response.status_code})")
                return False
                
            total_size = int(response.headers.get("Content-Length", 0))
            with open(filepath, "wb") as f, tqdm(
                total=total_size, unit="B", unit_scale=True, desc=filename, leave=False
            ) as bar:
                for chunk in response.iter_bytes(chunk_size=8192):
                    f.write(chunk)
                    bar.update(len(chunk))
        print(f"   [SUCCESS] Successfully downloaded {filename}")
        return True
    except Exception as e:
        print(f"   [ERROR] Exception occurred downloading {filename}: {e}")
        if os.path.exists(filepath):
            os.remove(filepath)
        return False

def main():
    parser = argparse.ArgumentParser(description="Download HMC Sleep Staging Database Subset from PhysioNet")
    parser.add_argument("--subjects", type=int, default=5, help="Number of subjects to download (default: 5)")
    args = parser.parse_args()

    os.makedirs(LOCAL_DIR, exist_ok=True)
    print(f"Saving recordings to: {LOCAL_DIR}")
    print(f"Target: Downloading data for {args.subjects} subjects...")

    # We will download subjects starting from SN001 up to the requested number
    # Format is SN001, SN002, ..., SN151
    with httpx.Client(timeout=60.0) as client:
        success_count = 0
        for i in range(1, args.subjects + 1):
            subject_str = f"SN{i:03d}"
            print(f"\nProcessing Subject {i}/{args.subjects} ({subject_str})...")
            
            psg_file = f"{subject_str}.edf"
            score_file = f"{subject_str}_sleepscoring.edf"
            
            psg_ok = download_file(client, psg_file)
            score_ok = download_file(client, score_file)
            
            if psg_ok and score_ok:
                success_count += 1
                
        print(f"\nFinished download process! Successfully downloaded {success_count}/{args.subjects} subjects.")

if __name__ == "__main__":
    main()

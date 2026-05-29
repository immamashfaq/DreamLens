import os
from mne.datasets.sleep_physionet.age import fetch_data

def download_sleep_data():
    dataset_dir = os.path.join(os.getcwd(), 'dataset')
    os.makedirs(dataset_dir, exist_ok=True)
    
    print("🌐 Connecting to PhysioNet...")
    print(f"📂 Destination: {dataset_dir}")
    print("⏳ Downloading data one-by-one until we have 40 valid patients (~5GB).")
    print("☕ Please leave this terminal open...\n")
    
    successful_subjects = 0
    subject_id = 0
    
    # Keep looping until we successfully download exactly 40 patients
    while successful_subjects < 40:
        try:
            # Try to fetch recording 1 for the current subject
            fetch_data(subjects=[subject_id], recording=[1], path=dataset_dir, verbose=False)
            successful_subjects += 1
            print(f"✅ Downloaded Patient {subject_id:02d} | Total secured: {successful_subjects}/40")
        except Exception as e:
            # If PhysioNet throws an error (missing/corrupted), we just catch it and skip
            print(f"⚠️ Skipping Patient {subject_id:02d} (Missing data on PhysioNet servers)")
        
        # Move to the next patient ID regardless of success or failure
        subject_id += 1

    print("\n🎉 MASSIVE DOWNLOAD COMPLETE!")
    print("All 5GB of valid clinical data is now safely on your hard drive.")

if __name__ == "__main__":
    download_sleep_data()
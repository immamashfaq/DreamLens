import os
import sys
# Reconfigure stdout to support UTF-8 on Windows
sys.stdout.reconfigure(encoding='utf-8')
import glob
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import mne
import numpy as np
from sklearn.model_selection import train_test_split
from models_architecture import DreamLens_Transformer

LOCAL_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset", "hmc-sleep-data")

def load_hmc_data():
    print(f"Scanning HMC directory: {LOCAL_DATA_DIR}")
    
    psg_files = sorted(glob.glob(os.path.join(LOCAL_DATA_DIR, "SN*.edf")))
    # Filter out sleepscoring files from the list of raw psg files
    psg_files = [f for f in psg_files if not f.endswith("_sleepscoring.edf")]
    
    if not psg_files:
        raise ValueError("No PSG files found. Make sure download_hmc.py has run successfully.")
        
    print(f"Found {len(psg_files)} clinical PSG subjects.")
    
    all_X = []
    all_y = []
    
    for idx, psg_path in enumerate(psg_files):
        subject_name = os.path.basename(psg_path).replace(".edf", "")
        score_path = psg_path.replace(".edf", "_sleepscoring.edf")
        
        if not os.path.exists(score_path):
            print(f"⚠️ Warning: Scoring file missing for {subject_name}. Skipping.")
            continue
            
        print(f"⏳ Loading Subject {idx+1}/{len(psg_files)}: {subject_name}...")
        try:
            # Load raw signals and annotations
            raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)
            annot = mne.read_annotations(score_path)
            raw.set_annotations(annot, emit_warning=False)
            
            # Select an central/midline EEG channel dynamically
            available_chans = raw.ch_names
            target_ch = None
            for ch in ['EEG C3-M2', 'EEG C4-M1', 'EEG C3-M2', 'EEG C4-M1']:
                if ch in available_chans:
                    target_ch = ch
                    break
            if not target_ch:
                for ch in available_chans:
                    if 'C3' in ch or 'C4' in ch or 'Cz' in ch:
                        target_ch = ch
                        break
            if not target_ch:
                target_ch = available_chans[0]
                
            print(f"   Selected channel: {target_ch}")
            raw.pick([target_ch])
            
            # Resample dynamically from 256 Hz to 100 Hz so 30s epoch is exactly 3000 samples
            raw.resample(100, verbose=False)
            raw.filter(0.3, 35.0, verbose=False)
            
            # Map labels to 0-4 range
            mapping = {
                'Sleep stage W': 0, 'W': 0, 'Wake': 0,
                'Sleep stage N1': 1, 'N1': 1, 'Stage 1': 1,
                'Sleep stage N2': 2, 'N2': 2, 'Stage 2': 2,
                'Sleep stage N3': 3, 'N3': 3, 'Stage 3': 3, 'Sleep stage N4': 3, 'N4': 3, 'Stage 4': 3,
                'Sleep stage R': 4, 'R': 4, 'REM': 4, 'Sleep stage REM': 4
            }
            
            # Extract 30-second epochs matching the event annotations
            events, event_id = mne.events_from_annotations(raw, event_id=mapping, chunk_duration=30., verbose=False)
            epochs = mne.Epochs(raw, events, event_id=event_id, tmin=0., tmax=30. - 1. / raw.info['sfreq'], baseline=None, preload=True, verbose=False)
            
            epoch_data = epochs.get_data() * 1e6 # convert to microvolts
            epoch_labels = epochs.events[:, 2]
            
            all_X.append(epoch_data)
            all_y.append(epoch_labels)
            print(f"   ✅ Successfully extracted {len(epoch_labels)} epochs.")
            
        except Exception as e:
            print(f"   ❌ Error loading {subject_name}: {e}")
            
    if not all_X:
        raise ValueError("Failed to load or parse epochs from any subject.")
        
    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    return X, y

def main():
    parser = argparse.ArgumentParser(description="Train DreamLens Transformer on HMC Sleep Data")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    args = parser.parse_args()

    # Determine Device (prefer CUDA for fast training if user has GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {device}")

    # Load data
    X, y = load_hmc_data()
    print(f"🎯 Total dataset shape: X={X.shape}, y={y.shape}")

    # Normalize data
    print("Normalizing EEG signals...")
    X = (X - np.mean(X)) / (np.std(X) + 1e-8)

    # Train/Val split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create PyTorch datasets
    X_train_t = torch.tensor(X_train.astype(np.float32))
    y_train_t = torch.tensor(y_train.astype(np.int64))
    X_val_t = torch.tensor(X_val.astype(np.float32))
    y_val_t = torch.tensor(y_val.astype(np.int64))

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=args.batch_size, shuffle=False)

    # Model configuration
    model = DreamLens_Transformer(num_classes=5, in_chans=1).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training Loop
    print("\nStarting Training Loop...")
    best_acc = 0.0

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        train_acc = 100 * correct / total
        
        # Validation Loop
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for val_inputs, val_labels in val_loader:
                val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                val_outputs = model(val_inputs)
                _, val_predicted = torch.max(val_outputs.data, 1)
                val_total += val_labels.size(0)
                val_correct += (val_predicted == val_labels).sum().item()
                
        val_acc = 100 * val_correct / val_total
        print(f"Epoch [{epoch+1}/{args.epochs}] - Loss: {running_loss/len(train_loader):.4f} - Train Acc: {train_acc:.2f}% - Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "dreamlens_transformer_v1.pth")
            print("   🌟 Model improved! Saving weights...")

    print("\nTraining Complete!")
    print(f"Best Validation Accuracy: {best_acc:.2f}% | Output Saved: 'dreamlens_transformer_v1.pth'")

if __name__ == "__main__":
    main()

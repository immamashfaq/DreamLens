import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import mne
import numpy as np
from sklearn.model_selection import train_test_split
from models_architecture import DreamLens_Transformer 

# --- 1. SETUP & HYPERPARAMETERS ---
# Path to where I stored the Physionet EDF files on my local drive
LOCAL_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset", "physionet-sleep-data")

EPOCHS = 20           
BATCH_SIZE = 256      # Using a larger batch size since the RTX GPU can handle it
LEARNING_RATE = 0.001 
MAX_SUBJECTS = 10     # Limiting subjects so I don't run out of RAM (16GB limit)

# Check if CUDA is available for faster training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Training on device: {device}")
if device.type == 'cpu':
    print("⚠️ WARNING: PyTorch isn't seeing your RTX GPU. Check your CUDA installation!")

# --- 2. DATA PREPARATION ---
def load_local_sleep_data():
    print(f"📂 Scanning local directory: {LOCAL_DATA_DIR}")
    
    # Finding all the PSG and Hypnogram pairs using glob
    psg_files = sorted(glob.glob(os.path.join(LOCAL_DATA_DIR, "*PSG.edf")))
    hyp_files = sorted(glob.glob(os.path.join(LOCAL_DATA_DIR, "*Hypnogram.edf")))
    
    if not psg_files or not hyp_files:
        print(f"DEBUG: Found {len(psg_files)} PSG files and {len(hyp_files)} Hypnograms.")
        raise ValueError("❌ No matching .edf pairs found! Ensure files end with 'PSG.edf' and 'Hypnogram.edf'.")

    all_X = []
    all_y = []
    
    # Only process up to the limit I set earlier
    files_to_process = min(len(psg_files), MAX_SUBJECTS)
    print(f"⏳ Extracting brainwaves from {files_to_process} subjects...")

    for i in range(files_to_process):
        try:
            # Loading the raw EEG and the expert labels
            raw = mne.io.read_raw_edf(psg_files[i], preload=True, verbose=False)
            annot = mne.read_annotations(hyp_files[i])
            raw.set_annotations(annot, emit_warning=False)

            # Filtering the signal and picking the Fpz-Cz channel (common in research)
            raw.pick(['EEG Fpz-Cz'])
            raw.filter(0.3, 35.0, verbose=False) 

            # Standardizing the labels. Merging 3 and 4 into a single "Deep Sleep" category
            mapping = {
                'Sleep stage W': 0, 
                'Sleep stage 1': 1, 
                'Sleep stage 2': 2, 
                'Sleep stage 3': 3, 
                'Sleep stage 4': 3, 
                'Sleep stage R': 4
            }
            
            # Splitting the continuous data into 30-second blocks
            events, event_id = mne.events_from_annotations(raw, event_id=mapping, chunk_duration=30., verbose=False)
            epochs = mne.Epochs(raw, events, event_id=event_id, tmin=0., tmax=30. - 1. / raw.info['sfreq'], baseline=None, preload=True, verbose=False)

            # Scaling to microvolts and storing
            all_X.append(epochs.get_data() * 1e6)
            all_y.append(epochs.events[:, 2])
            print(f"   ✅ Loaded Subject {i+1}: {psg_files[i].split(os.sep)[-1]}")
            
        except Exception as e:
            print(f"   ⚠️ Skipping subject {i+1} due to error: {e}")

    # Combine everything into one big dataset
    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    
    print(f"🎯 Total training epochs extracted: {len(y)}")
    return X, y

# Run the loader function
X, y = load_local_sleep_data()

# Standardization is crucial for Transformer stability
print("🧮 Normalizing EEG signals...")
X = (X - np.mean(X)) / (np.std(X) + 1e-8)

# Splitting 80% for training and 20% for checking performance
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors so the model can read them
X_train_t = torch.tensor(X_train.astype(np.float32))
y_train_t = torch.tensor(y_train.astype(np.int64))
X_val_t = torch.tensor(X_val.astype(np.float32))
y_val_t = torch.tensor(y_val.astype(np.int64))

# Setup data loaders for batch processing
train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=BATCH_SIZE, shuffle=False)

# --- 3. INITIALIZE MODEL ---
# Putting the model on the GPU
print("🤖 Initializing DreamLens Transformer...")
model = DreamLens_Transformer(num_classes=5, in_chans=1).to(device)
criterion = nn.CrossEntropyLoss() # Loss function for multi-class
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- 4. THE TRAINING LOOP ---
print("\n🔥 Starting High-Performance Training Loop...")
best_acc = 0.0

for epoch in range(EPOCHS):
    model.train() # Make sure layers like dropout are active
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Reset gradients, predict, backprop, and update weights
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        # Keep track of stats
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    train_acc = 100 * correct / total
    
    # Evaluation on validation set
    model.eval() 
    val_correct = 0
    val_total = 0
    with torch.no_grad(): # No need to track gradients here
        for val_inputs, val_labels in val_loader:
            val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
            val_outputs = model(val_inputs)
            _, val_predicted = torch.max(val_outputs.data, 1)
            val_total += val_labels.size(0)
            val_correct += (val_predicted == val_labels).sum().item()
            
    val_acc = 100 * val_correct / val_total
    
    print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {running_loss/len(train_loader):.4f} - Train Acc: {train_acc:.2f}% - Val Acc: {val_acc:.2f}%")
    
    # Save the model only if it's the best one so far
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "dreamlens_transformer_v1.pth")
        print("   🌟 Model improved! Saving weights...")

print("\n🎉 Training Complete!")
print(f"Final Weights: 'dreamlens_transformer_v1.pth' | Best Val Accuracy: {best_acc:.2f}%")
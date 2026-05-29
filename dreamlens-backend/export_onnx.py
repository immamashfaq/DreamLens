import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
import torch
from models_architecture import DreamLens_Transformer

def export_model():
    print("🤖 Initializing model...")
    model = DreamLens_Transformer(num_classes=5, in_chans=1)
    
    weights_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dreamlens_transformer_v1.pth")
    print(f"📦 Loading weights from: {weights_path}")
    
    device = torch.device("cpu")
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()
    
    # Standard dummy input: (batch_size, in_chans, seq_len)
    # Physionet standard is 100Hz, so 30 seconds = 3000 samples
    dummy_input = torch.randn(1, 1, 3000, dtype=torch.float32)
    
    onnx_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dreamlens_transformer.onnx")
    print(f"🚀 Exporting to ONNX format at: {onnx_path}")
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size', 2: 'seq_len'}, 'output': {0: 'batch_size'}}
    )
    print("✅ Model successfully exported to ONNX format!")
    
    # Merge external weights data back into the main .onnx file to make it self-contained
    import onnx
    print("📦 Embedding external weights into a single self-contained ONNX file...")
    loaded_model = onnx.load(onnx_path)
    onnx.save_model(loaded_model, onnx_path, save_as_external_data=False)
    
    data_file_path = onnx_path + ".data"
    if os.path.exists(data_file_path):
        os.remove(data_file_path)
        print("🗑️ Cleaned up external data file (.data)")

if __name__ == "__main__":
    export_model()

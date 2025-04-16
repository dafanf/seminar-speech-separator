import torch
import torchaudio
import os
from model import ConvTasNet  # Pastikan class ConvTasNet sesuai dengan modelmu

# === SETUP ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# === LOAD MODEL ===
model_path = "models\conv_tasnet_2.pth"
model = ConvTasNet().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print(f"[INFO] Loaded model from {model_path}")

# === INFERENCE FUNCTION ===
def separate_audio(input_path, output_path):
    waveform, sr = torchaudio.load(input_path)
    
    # Pastikan mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    input_tensor = waveform.unsqueeze(0).to(device)  # [1, 1, T]

    with torch.no_grad():
        estimated = model(input_tensor).squeeze(0).cpu()  # [1, T]

    # Simpan hasil
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torchaudio.save(output_path, estimated, sr)
    print(f"[✓] {os.path.basename(output_path)} saved")

# === BATCH PROCESSING ===
def run_inference_batch(mixture_dir, output_dir="inference_outputs"):
    os.makedirs(output_dir, exist_ok=True)
    mixture_files = sorted([f for f in os.listdir(mixture_dir) if f.endswith(".wav")])

    for fname in mixture_files:
        mix_path = os.path.join(mixture_dir, fname)
        out_path = os.path.join(output_dir, fname)
        separate_audio(mix_path, out_path)

# === RUN ===
if __name__ == "__main__":
    mixture_dir = "E:\\POLBAN\\Semester-8\\TA\\Dataset\\Ready\\Coba\\16k\\Split\\Test\\Mixture-2Speaker"
    output_dir = "inference_results"
    run_inference_batch(mixture_dir, output_dir)

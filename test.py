import torch
import torchaudio
from pesq import pesq
from scipy.signal import resample
from model import ConvTasNet  # pastikan ini sesuai dengan struktur kamu
import os

# --- Parameter ---
sample_rate = 16000
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Load model ---
model = ConvTasNet().to(device)
model.load_state_dict(torch.load("models/conv_tasnet_2.pth", map_location=device))
model.eval()

# --- Load audio ---
def load_audio(path):
    waveform, sr = torchaudio.load(path)
    if sr != sample_rate:
        waveform = torchaudio.transforms.Resample(sr, sample_rate)(waveform)
    return waveform

# --- Hitung PESQ ---
def compute_pesq(ref, deg):
    # PESQ expects numpy arrays, 16 kHz, mono
    return pesq(sample_rate, ref.squeeze().cpu().numpy(), deg.squeeze().cpu().numpy(), 'wb')

# --- Evaluasi satu file ---
def evaluate_one(mixture_path, target_path):
    mixture = load_audio(mixture_path).to(device)
    target = load_audio(target_path).to(device)

    with torch.no_grad():
        estimate = model(mixture)

    # Normalize panjang
    min_len = min(target.shape[-1], estimate.shape[-1])
    target = target[..., :min_len]
    estimate = estimate[..., :min_len]

    pesq_score = compute_pesq(target, estimate)
    print(f'PESQ Score: {pesq_score:.3f}')

# --- Contoh penggunaan ---
if __name__ == "__main__":
    mixture_file = "E:\POLBAN\Semester-8\TA\Dataset\Ready\Coba\\16k\Split\Train\Mixture-2Speaker\p11_0001-p12_0059.wav"
    target_file = "E:\POLBAN\Semester-8\TA\Dataset\Ready\Coba\\16k\Split\Train\Single-Speech\p11_0001.wav"  # asumsi kamu mau evaluasi speaker 1

    evaluate_one(mixture_file, target_file)

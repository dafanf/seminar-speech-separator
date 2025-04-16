import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchaudio
import os
from pesq import pesq
import glob
import matplotlib.pyplot as plt
import json
import csv
import time
import ast
import pandas as pd

# === SETUP DEVICE ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")
if torch.cuda.is_available():
    print(f"[INFO] CUDA Device Name: {torch.cuda.get_device_name(0)}")

# ======= Dataset Loader =======
class SpeechSeparationDataset(Dataset):
    def __init__(self, csv_file):
        print(f"[INFO] Loading dataset from {csv_file}...")
        self.data = pd.read_csv(csv_file)
        print(f"[INFO] Loaded {len(self.data)} samples.")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        mixture_path = self.data.iloc[idx, 0]
        target_paths = ast.literal_eval(self.data.iloc[idx, 1])  # ubah string list jadi list Python

        mixture, sr = torchaudio.load(mixture_path)

        targets = []
        for path in target_paths:
            wav, _ = torchaudio.load(path)
            targets.append(wav)

        min_len = min(wav.shape[1] for wav in targets + [mixture])
        mixture = mixture[:, :min_len]
        targets = [wav[:, :min_len] for wav in targets]
        targets = torch.stack(targets, dim=0).squeeze(1)  # (n_speakers, T)

        return mixture, targets, sr, mixture_path

# ======= Function Collate for Padding =======
def collate_fn(batch):
    mixtures, targets, srs, paths = zip(*batch)
    max_len = max(m.shape[1] for m in mixtures)

    padded_mixtures = [torch.nn.functional.pad(m, (0, max_len - m.shape[1])) for m in mixtures]
    padded_targets = [torch.nn.functional.pad(t, (0, max_len - t.shape[1])) for t in targets]

    return torch.stack(padded_mixtures), torch.stack(padded_targets), srs[0], paths


# ======= Encoder-Decoder from Conv-TasNet =======
class Encoder(nn.Module):
    def __init__(self, kernel_size=16, stride=8, num_filters=512):
        super(Encoder, self).__init__()
        self.conv = nn.Conv1d(1, num_filters, kernel_size, stride=stride, bias=False)
    
    def forward(self, x):
        return torch.relu(self.conv(x))

class Decoder(nn.Module):
    def __init__(self, kernel_size=16, stride=8, num_filters=512):
        super(Decoder, self).__init__()
        self.deconv = nn.ConvTranspose1d(num_filters, 1, kernel_size, stride=stride, bias=False)
    
    def forward(self, x):
        return self.deconv(x)

# ======= Separator (TCN) =======
class TCN(nn.Module):
    def __init__(self, num_layers=8, num_channels=512):
        super(TCN, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(nn.Conv1d(num_channels, num_channels, kernel_size=3, dilation=2**i, padding=2**i))
            layers.append(nn.ReLU())
        self.tcn = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.tcn(x)

# ======= Conv-TasNet Model =======
class ConvTasNet(nn.Module):
    def __init__(self):
        super(ConvTasNet, self).__init__()
        self.encoder = Encoder()
        self.separator = TCN()
        self.decoder = Decoder()
    
    def forward(self, x):
        encoded = self.encoder(x)
        separated = self.separator(encoded)
        decoded = self.decoder(separated)
        return decoded

# ======= Function to calculate PESQ =======
def compute_metrics(clean, enhanced, sr):
    # Jika sinyal memiliki lebih dari 1 dimensi, ambil channel pertama.
    if clean.dim() > 1:
        clean = clean[0]
    if enhanced.dim() > 1:
        enhanced = enhanced[0]
    
    clean_np = clean.cpu().detach().numpy()
    enhanced_np = enhanced.cpu().detach().numpy()
    
    try:
        pesq_score = pesq(sr, clean_np, enhanced_np, 'wb')
    except Exception as e:
        print(f'[WARNING] PESQ computation failed: {e}')
        pesq_score = 0.0
    
    return pesq_score


# ======= Discriminator MetricGAN =======
class MetricDiscriminator(nn.Module):
    def __init__(self):
        super(MetricDiscriminator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=15, kernel_size=(5, 5))
        self.conv2 = nn.Conv2d(in_channels=15, out_channels=25, kernel_size=(7, 7))
        self.conv3 = nn.Conv2d(in_channels=25, out_channels=40, kernel_size=(9, 9))
        self.conv4 = nn.Conv2d(in_channels=40, out_channels=50, kernel_size=(11, 11))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(50, 50)
        self.fc2 = nn.Linear(50, 10)
        self.fc3 = nn.Linear(10, 1)
        self.activation = nn.LeakyReLU()

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))
        x = self.pool(x).view(x.size(0), -1)  # Flatten
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        return self.fc3(x)

def prepare_discriminator_input(clean_waveform, enhanced_waveform, n_fft=512, hop_length=128):
    """
    Mengembalikan tensor bentuk (B, 2, F, T) yang berisi magnitude spectrogram clean dan enhanced.
    Jika input memiliki lebih dari 2 dimensi (misalnya [B, n_speakers, T]), 
    gunakan channel pertama (misalnya, speaker pertama) untuk perhitungan STFT.
    """
    # Jika tensor memiliki dimensi > 2, ambil channel pertama.
    if clean_waveform.dim() > 2:
        clean_waveform = clean_waveform[:, 0, :]
    if enhanced_waveform.dim() > 2:
        enhanced_waveform = enhanced_waveform[:, 0, :]

    # clean_waveform dan enhanced_waveform sekarang bentuknya [B, T]
    window = torch.hann_window(n_fft).to(clean_waveform.device)

    clean_spec = torch.abs(torch.stft(
        clean_waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        return_complex=True
    ))

    enhanced_spec = torch.abs(torch.stft(
        enhanced_waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        return_complex=True
    ))

    # Sesuaikan dimensi spektrum sehingga sama ukuran (F, T)
    min_freq = min(clean_spec.shape[-2], enhanced_spec.shape[-2])
    min_time = min(clean_spec.shape[-1], enhanced_spec.shape[-1])
    clean_spec = clean_spec[..., :min_freq, :min_time]
    enhanced_spec = enhanced_spec[..., :min_freq, :min_time]

    # Gabungkan kedua spektrum di dimensi channel, menghasilkan tensor (B, 2, F, T)
    x = torch.stack([enhanced_spec, clean_spec], dim=1)
    return x


# Save model checkpoint
def save_checkpoint(epoch, model_S, model_D, optimizer_S, optimizer_D, loss_S, loss_D):
    checkpoint = {
        'epoch': epoch,
        'model_S_state_dict': model_S.state_dict(),
        'model_D_state_dict': model_D.state_dict(),
        'optimizer_S_state_dict': optimizer_S.state_dict(),
        'optimizer_D_state_dict': optimizer_D.state_dict(),
        'loss_S': loss_S,
        'loss_D': loss_D
    }
    directory = "checkpoints"
    if not os.path.exists(directory):
        os.makedirs(directory)
    checkpoint_path = os.path.join(directory, f"checkpoint_epoch_{epoch+1}.pth")
    torch.save(checkpoint, checkpoint_path)
    print(f"[INFO] Checkpoint saved to {checkpoint_path}")

def find_latest_checkpoint():
    directory = "checkpoints"
    checkpoint_files = glob.glob(os.path.join(directory, "checkpoint_epoch_*.pth"))
    if not checkpoint_files:
        return None
    latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split("_")[-1].split(".")[0]))
    return latest_checkpoint

def load_checkpoint(model_S, model_D, optimizer_S, optimizer_D):
    checkpoint_path = find_latest_checkpoint()
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model_S.load_state_dict(checkpoint['model_S_state_dict'])
        model_D.load_state_dict(checkpoint['model_D_state_dict'])
        optimizer_S.load_state_dict(checkpoint['optimizer_S_state_dict'])
        optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"[INFO] Checkpoint {checkpoint_path} loaded, resuming from epoch {start_epoch}")
        return start_epoch
    else:
        print("[INFO] No checkpoint found, starting training from scratch.")
        return 0

def evaluate(model, train_loader):
    model.eval() 
    total_pesq, count = 0, 0

    with torch.no_grad():
        for mixture, target, sr, _ in train_loader:
            mixture = mixture.to(device)
            target = target.to(device)
            output = model(mixture)
            pesq_score = compute_metrics(target[0], output[0], sr)
            total_pesq += pesq_score
            count += 1

    avg_pesq = total_pesq / count
    print(f"[VALIDATION] PESQ: {avg_pesq:.3f}")
    return avg_pesq

def plot_pesq_curve(pesq_history):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(pesq_history) + 1), pesq_history, marker='o', linestyle='-')
    plt.xlabel("Epoch")
    plt.ylabel("PESQ Score")
    plt.title("Training Progress: PESQ vs Epoch")
    plt.grid(True)
    plt.savefig("pesq_training_curve.png")
    print("[INFO] PESQ curve saved as 'pesq_training_curve.png'.")
    plt.show()


def save_pesq_history(pesq_history, filename="pesq_history.json"):
    with open(filename, "w") as f:
        json.dump(pesq_history, f)
    print(f"[INFO] PESQ history saved to {filename}")

def load_pesq_history(filename="pesq_history.json"):
    if os.path.exists(filename):
        with open(filename, "r") as f:
            return json.load(f)
    return [] 

def init_csv_logger(filepath="training_log.csv"):
    if not os.path.exists(filepath):
        with open(filepath, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                "Epoch", "Loss_Discriminator", "Loss_Separator", 
                "Avg_PESQ", "Validation_PESQ", "Duration_Minutes"
            ])
    return filepath

def log_to_csv(filepath, epoch, loss_D, loss_S, pesq, validation, duration):
    with open(filepath, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([epoch, loss_D, loss_S, pesq, validation, duration])


def train(model_S, model_D, train_loader, val_loader, optimizer_S, optimizer_D, epochs, start_epoch=0):
    print("[INFO] Starting training...")
    criterion = nn.MSELoss()
    pesq_history = load_pesq_history()
    csv_path = init_csv_logger()

    for epoch in range(start_epoch, epochs): 
        print(f"[INFO] Epoch {epoch+1}/{epochs}...")
        epoch_start_time = time.time()
        total_pesq, count = 0, 0

        for batch_idx, (mixture, target, sr, mixture_path) in enumerate(train_loader):
            print(f"[INFO] Processing batch {batch_idx+1}/{len(train_loader)}")
            mixture, target = mixture.to(device), target.to(device)

            # Forward pass separator
            fake = model_S(mixture)

            # Prepare discriminator inputs
            D_real_input = prepare_discriminator_input(target, target).to(device)  # target vs target (label 1)
            D_fake_input = prepare_discriminator_input(target, fake.detach()).to(device)  # enhanced vs target (label 0)

            # === Train Discriminator ===
            optimizer_D.zero_grad()
            real_score = model_D(D_real_input)
            fake_score = model_D(D_fake_input)
            loss_D = criterion(real_score, torch.ones_like(real_score)) + criterion(fake_score, torch.zeros_like(fake_score))
            loss_D.backward()
            optimizer_D.step()

            # === Train Separator ===
            optimizer_S.zero_grad()
            D_fake_for_S = prepare_discriminator_input(target, fake).to(device)
            fake_score = model_D(D_fake_for_S)
            loss_S = criterion(fake_score, torch.ones_like(fake_score))
            loss_S.backward()
            optimizer_S.step()

            # === Metrics ===
            pesq_val = compute_metrics(target[0], fake[0], sr)
            total_pesq += pesq_val
            count += 1

        epoch_duration = time.time() - epoch_start_time
        avg_pesq = total_pesq / count
        pesq_history.append(avg_pesq) 
        save_pesq_history(pesq_history)
        print(f"Epoch {epoch+1}/{epochs}, Loss_D: {loss_D.item():.4f}, Loss_S: {loss_S.item():.4f}, PESQ: {avg_pesq:.3f}")

        if (epoch + 1) % 5 == 0:
            save_checkpoint(epoch, model_S, model_D, optimizer_S, optimizer_D, loss_S, loss_D)
        eval = evaluate(model_S, val_loader)
        log_to_csv(csv_path, epoch+1, loss_D.item(), loss_S.item(), avg_pesq, eval, epoch_duration / 60)

    print("[INFO] Saving trained models...")
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model_S.state_dict(), os.path.join(model_dir, "conv_tasnet.pth"))
    torch.save(model_D.state_dict(), os.path.join(model_dir, "discriminator.pth"))
    print("[INFO] Models saved successfully.")
    plot_pesq_curve(pesq_history)



# ======= Training Execution =======
if __name__ == "__main__":
    print("[INFO] Initializing models...")
    # Hyperparameters
    epochs = 2
    lr = 0.001
    batch_size = 8

    model_S = ConvTasNet().to(device)
    model_D = MetricDiscriminator().to(device)
    # Update dataset paths to reflect the new speaker subfolders
    train_dataset = SpeechSeparationDataset(csv_file='data/train.csv')
    val_dataset = SpeechSeparationDataset(csv_file='data/valid.csv')


    optimizer_S = optim.Adam(model_S.parameters(), lr=lr)
    optimizer_D = optim.Adam(model_D.parameters(), lr=lr * 0.5)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    print("[INFO] DataLoader ready. Starting training...")

    start_epoch = load_checkpoint(model_S, model_D, optimizer_S, optimizer_D)
    train(model_S, model_D, train_loader, val_loader, optimizer_S, optimizer_D, epochs, start_epoch)
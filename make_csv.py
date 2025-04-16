import os
import csv

def make_csv(mixture_dir, single_dir, output_csv):
    rows = []
    for filename in os.listdir(mixture_dir):
        if not filename.endswith(".wav"):
            continue
        mixture_path = os.path.relpath(os.path.join(mixture_dir, filename))

        # Ambil nama 2 source dari nama file
        try:
            s1_name, s2_name = filename.replace(".wav", "").split("-")
        except ValueError:
            print(f"Skipping malformed file name: {filename}")
            continue

        s1_path = os.path.relpath(os.path.join(single_dir, s1_name + ".wav"))
        s2_path = os.path.relpath(os.path.join(single_dir, s2_name + ".wav"))

        if not os.path.exists(s1_path) or not os.path.exists(s2_path):
            print(f"Source file missing for {filename}")
            continue

        sources = [s1_path, s2_path]
        rows.append((mixture_path, str(sources)))

    # Tulis CSV
    with open(output_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['mixture', 'sources'])
        writer.writerows(rows)

    print(f"CSV saved to: {output_csv}")


# === Jalankan untuk datasetmu ===
make_csv(
    mixture_dir="data/Train/Mixture-2Speaker",
    single_dir="data/Train/Single-Speech",
    output_csv="train.csv"
)

make_csv(
    mixture_dir="data/Validation/Mixture-2Speaker",
    single_dir="data/Validation/Single-Speech",
    output_csv="valid.csv"
)

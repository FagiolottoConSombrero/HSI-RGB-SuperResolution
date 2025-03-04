import os
import torch
import numpy as np
import tifffile as tiff
from PIL import Image
from mst import *

# Definizione delle cartelle
input_folder = "/home/ubuntu/Flowers/flowers_hsi/"
output_folder = "/home/ubuntu/Flowers/flowers_hsi_mst/"  # Nuova cartella per le immagini processate

# Creazione della cartella di output se non esiste
os.makedirs(output_folder, exist_ok=True)

# Controlla se è disponibile la GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Utilizzo del dispositivo: {device}")

# Carica il modello MST_Plus_Plus sulla GPU (se disponibile)
model = MST_Plus_Plus().to(device)
checkpoint = torch.load('/home/ubuntu/HSI-RGB-SuperResolution/model_weights$/model_weights/mst_plus_plus.pth',
                        map_location=device)
model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}, strict=True)
model.eval()  # Modalità valutazione

# Ottieni la lista delle immagini nella cartella di input, escludendo file che iniziano con "._"
image_files = [f for f in os.listdir(input_folder)
               if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')) and not f.startswith("._")]

# Processa ogni immagine
for image_file in image_files:
    # Carica l'immagine
    img_path = os.path.join(input_folder, image_file)
    img = Image.open(img_path).convert('RGB')  # Carica immagine e converte in RGB

    # Converti in array NumPy e poi in tensore PyTorch
    img_np = np.array(img)
    img_tensor = torch.from_numpy(img_np).float().permute(2, 0, 1).unsqueeze(0) / 255.0  # Normalizza tra 0 e 1

    # Sposta il tensore sulla GPU
    img_tensor = img_tensor.to(device)

    # Genera l'immagine HSI con il modello sulla GPU
    with torch.no_grad():
        hsi = model(img_tensor)  # Output del modello

    # Riporta l'output sulla CPU e mantieni il formato originale (C, H, W)
    hsi_np = hsi.squeeze(0).cpu().numpy()  # Shape: (31, H, W)

    # Definisci il percorso di salvataggio nella cartella `flowers_hsi_mst`
    hsi_path = os.path.join(output_folder, image_file).replace(".png", ".tiff").replace(".jpg", ".tiff")

    # Salva in formato TIFF per non perdere le bande spettrali
    tiff.imwrite(hsi_path, hsi_np.astype(np.float32))  # Usa float32 per mantenere i dati
    print(f"Immagine HSI salvata con 31 bande: {hsi_path}")

print("Elaborazione completata per tutte le immagini.")

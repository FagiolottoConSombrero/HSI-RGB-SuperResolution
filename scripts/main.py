import h5py
import numpy as np
import json
import matplotlib.pyplot as plt
from hdf5 import *
from mst import *
import cv2
import os
from PIL import Image


if __name__ == '__main__':
    # Percorsi delle cartelle
    input_folder = "/Volumes/Lexar/Flowers/flowers_hsi/"

    # Controlla se è disponibile la GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilizzo del dispositivo: {device}")

    # Carica il modello MST_Plus_Plus sulla GPU (se disponibile)
    model = MST_Plus_Plus().to(device)
    checkpoint = torch.load('/Users/kolyszko/PycharmProjects/HSI-RgbSR/model_weights/mst_plus_plus.pth',
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

        # Riporta l'output sulla CPU e lo converte in immagine
        hsi_np = (hsi.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        hsi_img = Image.fromarray(hsi_np)  # Converti in immagine PIL

        # Sovrascrive il file originale
        hsi_img.save(img_path)

        print(f"Immagine elaborata e salvata: {img_path}")

    print("Elaborazione completata per tutte le immagini.")













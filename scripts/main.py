import numpy as np
from mst import *
import os
from PIL import Image
import tifffile as tiff


if __name__ == '__main__':
    # Percorsi delle cartelle
    input_folder = "/home/ubuntu/Flowers/flowers_hsi/"

    # Controlla se è disponibile la GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilizzo del dispositivo: {device}")

    # Carica il modello MST_Plus_Plus sulla GPU (se disponibile)
    model = MST_Plus_Plus().to(device)
    checkpoint = torch.load('/home/ubuntu/HSI-RGB-SuperResolution/model_weights/mst_plus_plus.pth',
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
            hsi = model(img_tensor)  # Output del modello (potenzialmente con più canali)

        # Riporta l'output sulla CPU
        hsi_np = (hsi.squeeze(0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

        # Percorso per il file TIFF
        tiff_path = img_path.replace('.png', '.tiff')

        # Salva l'immagine HSI in formato TIFF multi-canale
        tiff.imwrite(tiff_path, hsi_np)

        print(f"Immagine HSI salvata in TIFF: {tiff_path}")

    print("Elaborazione completata per tutte le immagini.")













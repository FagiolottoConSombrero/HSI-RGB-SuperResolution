import numpy as np
from mst import *
import os
from PIL import Image
import tifffile as tiff
import cv2


if __name__ == '__main__':
    # Percorsi delle cartelle
    input_folder = "/home/ubuntu/Flowers/flowers_hsi/"  # Cartella con gli HSI HR (TIFF)
    output_folder = "/home/ubuntu/Flowers/flowers_hsi_LR/"  # Cartella per le versioni LR

    # Creazione della cartella di output se non esiste
    os.makedirs(output_folder, exist_ok=True)

    # Ottieni la lista delle immagini TIFF nella cartella di input
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.tiff') and not f.startswith("._")]

    # Parametri del filtro Gaussiano
    kernel_size = (8, 8)  # µ = 8
    sigma = 3  # σ = 3

    # Processa ogni immagine
    for image_file in image_files:
        print(f"Processing: {image_file}")

        # Carica l'immagine HSI multi-canale
        img_path = os.path.join(input_folder, image_file)
        hsi = tiff.imread(img_path).astype(np.float32)  # Carica in float32 per evitare perdita di precisione

        # Applica il blur gaussiano a ogni canale spettrale
        hsi_blurred = np.zeros_like(hsi)
        for i in range(hsi.shape[2]):  # Per ogni banda spettrale
            hsi_blurred[:, :, i] = cv2.GaussianBlur(hsi[:, :, i], kernel_size, sigma)

        # Downsampling x4 usando interpolazione bilineare (puoi anche provare bicubica)
        hsi_LR = cv2.resize(hsi_blurred, (hsi.shape[1] // 4, hsi.shape[0] // 4), interpolation=cv2.INTER_LINEAR)

        # Salva l'immagine ridotta nella nuova cartella
        output_path = os.path.join(output_folder, image_file)
        tiff.imwrite(output_path, hsi_LR.astype(np.uint8))  # Converti in uint8 per il salvataggio

        print(f"Salvato LR HSI: {output_path}")

    print("Tutte le immagini LR HSI sono state generate!")













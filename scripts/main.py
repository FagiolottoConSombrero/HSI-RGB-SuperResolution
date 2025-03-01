import numpy as np
from mst import *
import os
from PIL import Image
import tifffile as tiff
import cv2


if __name__ == '__main__':
    input_folder = "/home/ubuntu/Flowers/flowers_hsi/"  # HSI originale (HR)
    output_folder_LR = "/home/ubuntu/Flowers/flowers_hsi_LR/"  # HSI ridotto
    output_folder_upsampled = "/home/ubuntu/Flowers/flowers_hsi_upsampled/"  # HSI riportato a HR
    output_folder_rgb = "/home/ubuntu/Flowers/flowers_hsi_rgb/"  # RGB finale

    # 🛠 Crea le cartelle di output se non esistono
    os.makedirs(output_folder_LR, exist_ok=True)
    os.makedirs(output_folder_upsampled, exist_ok=True)
    os.makedirs(output_folder_rgb, exist_ok=True)

    # Carica le curve di sensibilità della Nikon D700 (3x31)
    srf = np.array([
        [0.005, 0.007, 0.012, 0.015, 0.023, 0.025, 0.030, 0.026, 0.024, 0.019, 0.010, 0.004, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0.000, 0.000, 0.000, 0.000, 0.000, 0.001, 0.002, 0.003, 0.005, 0.007, 0.012, 0.013, 0.015, 0.016, 0.017, 0.02,
         0.013, 0.011, 0.009, 0.005, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.002, 0.002, 0.003],
        [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
         0.001, 0.003, 0.010, 0.012, 0.013, 0.022, 0.020, 0.020, 0.018, 0.017, 0.016, 0.016, 0.014, 0.014, 0.013]
    ])

    # 🔍 Ottieni la lista delle immagini HSI nella cartella di input
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.tiff') and not f.startswith("._")]

    # 🎯 Parametri del filtro Gaussiano per la riduzione
    kernel_size = (8, 8)
    sigma = 3  # σ = 3

    # 🚀 Processa ogni immagine
    for image_file in image_files:
        print(f"Processing: {image_file}")

        # 📥 Carica l'immagine HSI originale (HR) nel formato (31, H, W)
        img_path = os.path.join(input_folder, image_file)
        hsi = tiff.imread(img_path).astype(np.float32)

        # ✅ 1️⃣ Converti in formato (H, W, 31) per compatibilità con OpenCV
        hsi = np.transpose(hsi, (1, 2, 0))  # Da (31, H, W) → (H, W, 31)

        # 📏 Ottieni altezza e larghezza
        h, w = hsi.shape[:2]
        new_w = max(w // 4, 1)  # Evita dimensioni 0
        new_h = max(h // 4, 1)

        # 🔽 2️⃣ **Gaussian Blur + Downsampling x4**
        hsi_blurred = np.zeros_like(hsi)
        for i in range(hsi.shape[2]):  # Applica il blur su ogni banda
            hsi_blurred[:, :, i] = cv2.GaussianBlur(hsi[:, :, i], kernel_size, sigma)

        # 🔽 Downsampling bilineare x4
        hsi_LR = np.zeros((new_h, new_w, hsi.shape[2]), dtype=np.float32)
        for i in range(hsi.shape[2]):
            hsi_LR[:, :, i] = cv2.resize(hsi_blurred[:, :, i], (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # 💾 Salva l'HSI ridotto (LR) in formato (31, H, W)
        lr_path = os.path.join(output_folder_LR, image_file)
        tiff.imwrite(lr_path, np.transpose(hsi_LR, (2, 0, 1)).astype(np.uint8))
        print(f"✅ Salvato LR HSI: {lr_path}")

        # 🔼 3️⃣ **Upsampling x4 per riportare l'immagine alla dimensione originale**
        hsi_HR = np.zeros((h, w, hsi.shape[2]), dtype=np.float32)
        for i in range(hsi.shape[2]):  # Upsampling bicubico per ogni banda
            hsi_HR[:, :, i] = cv2.resize(hsi_LR[:, :, i], (w, h), interpolation=cv2.INTER_CUBIC)

        # 💾 Salva l'HSI riportato a HR
        upsampled_path = os.path.join(output_folder_upsampled, image_file)
        tiff.imwrite(upsampled_path, np.transpose(hsi_HR, (2, 0, 1)).astype(np.uint8))
        print(f"✅ Salvato Upsampled HSI: {upsampled_path}")

        # 🎨 4️⃣ **Proiezione HSI → RGB con curve Nikon D700**
        assert hsi_HR.shape[2] == srf.shape[1], f"Errore: bande HSI ({hsi_HR.shape[2]}) ≠ curve Nikon ({srf.shape[1]})!"

        # 🔄 Proiezione usando le curve di sensibilità
        rgb_image = np.dot(hsi_HR.reshape(-1, hsi_HR.shape[2]), srf.T)  # (H*W, 31) @ (31, 3) → (H*W, 3)
        rgb_image = rgb_image.reshape(hsi_HR.shape[0], hsi_HR.shape[1], 3)  # (H, W, 3)

        # 🎚 Normalizza tra 0-255 e converte in uint8
        rgb_image = (rgb_image / np.max(rgb_image) * 255).clip(0, 255).astype(np.uint8)

        # 💾 Salva l'immagine RGB finale
        rgb_path = os.path.join(output_folder_rgb, image_file.replace('.tiff', '.png'))
        cv2.imwrite(rgb_path, cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))  # OpenCV usa BGR, quindi riconverti
        print(f"✅ Immagine RGB salvata: {rgb_path}")

    print("🚀 Tutto completato con successo!")













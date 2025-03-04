import os
from torch.utils.data import Dataset
from PIL import Image
import tifffile as tiff


class ImageDataset(Dataset):
    def __init__(self, image_paths='/home/ubuntu/Flowers/', transform=None):
        self.image_paths = image_paths
        self.transform = transform

        # Percorsi delle cartelle
        self.lr_hsi = os.path.join(self.image_paths, 'flowers_hsi_upsampled/')
        self.hr_rgb = os.path.join(self.image_paths, 'flowers_rgb/')
        self.lr_rgb = os.path.join(self.image_paths, 'flowers_hsi_rgb/')
        self.hr_hsi = os.path.join(self.image_paths, 'flowers_hsi/')

        # Ottenere i file in ogni cartella
        self.lr_hsi_files = sorted([f for f in os.listdir(self.lr_hsi) if f.endswith('.tiff')])
        self.hr_hsi_files = sorted([f for f in os.listdir(self.hr_hsi) if f.endswith('.tiff')])
        self.lr_rgb_files = sorted([f for f in os.listdir(self.lr_rgb) if f.endswith('.png')])
        self.hr_rgb_files = sorted([f for f in os.listdir(self.hr_rgb) if f.endswith('.png')])

        # Stampa di debug per verificare i file trovati
        print(f"📂 Cartella LR HSI: {self.lr_hsi} - {len(self.lr_hsi_files)} file trovati")
        print(f"📂 Cartella HR HSI: {self.hr_hsi} - {len(self.hr_hsi_files)} file trovati")
        print(f"📂 Cartella LR RGB: {self.lr_rgb} - {len(self.lr_rgb_files)} file trovati")
        print(f"📂 Cartella HR RGB: {self.hr_rgb} - {len(self.hr_rgb_files)} file trovati")

        # Se una lista è vuota, solleva un errore
        if not self.lr_hsi_files:
            raise ValueError(f"❌ Errore: Nessun file trovato nella cartella {self.lr_hsi}")

        if not self.hr_hsi_files:
            raise ValueError(f"❌ Errore: Nessun file trovato nella cartella {self.hr_hsi}")

        if not self.lr_rgb_files:
            raise ValueError(f"❌ Errore: Nessun file trovato nella cartella {self.lr_rgb}")

        if not self.hr_rgb_files:
            raise ValueError(f"❌ Errore: Nessun file trovato nella cartella {self.hr_rgb}")

    def __len__(self):
        return len(self.lr_hsi_files)

    def __getitem__(self, index):
        """
        Carica le immagini dal dataset e le trasforma se necessario.

        :param index: Indice dell'immagine
        :return: Dizionario contenente le immagini LR_HSI, HR_HSI, LR_RGB e HR_RGB
        """
        # Percorsi delle immagini
        lr_hsi_path = os.path.join(self.lr_hsi, self.lr_hsi_files[index])
        hr_hsi_path = os.path.join(self.hr_hsi, self.hr_hsi_files[index])
        lr_rgb_path = os.path.join(self.lr_rgb, self.lr_rgb_files[index])
        hr_rgb_path = os.path.join(self.hr_rgb, self.hr_rgb_files[index])

        # Caricare immagini HSI (TIFF)
        lr_hsi = tiff.imread(lr_hsi_path)
        hr_hsi = tiff.imread(hr_hsi_path)

        # Caricare immagini RGB (PNG)
        lr_rgb = Image.open(lr_rgb_path).convert("RGB")  # Converte in RGB
        hr_rgb = Image.open(hr_rgb_path).convert("RGB")  # Converte in RGB

        # Applicare trasformazioni se specificate
        if self.transform:
            lr_hsi = self.transform(lr_hsi)
            hr_hsi = self.transform(hr_hsi)
            lr_rgb = self.transform(lr_rgb)
            hr_rgb = self.transform(hr_rgb)

        return lr_hsi, hr_hsi, lr_rgb, hr_rgb

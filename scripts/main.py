import argparse
from utils.utils import *
from utils.dataset import ImageDataset
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description='Super Resolution Color Guided')
parser.add_argument('--batch_size', type=int, default='32', help='Training batch size')
parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train for")
parser.add_argument("--lr", type=float, default=0.0001, help="Learning Rate. Default=0.001")
parser.add_argument('--save_path', type=str,
                    default='', help="Path to model checkpoint")
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                    help="Device to run the script on: 'cuda' or 'cpu'. ")


def main():
    opt = parser.parse_args()
    print(opt)

    print("===> Loading data")
    dataset = ImageDataset()
    train_dataset, val_dataset, test_dataset = split_dataset(dataset)
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False)

    batch = next(iter(train_loader))

    # Se il DataLoader restituisce una tupla (più immagini per campione)
    if isinstance(batch, tuple) or isinstance(batch, list):
        for i, img in enumerate(batch):
            print(f"🔍 Shape dell'immagine {i + 1}: {img.shape}")


if __name__ == '__main__':
    main()

import torch
from tqdm import tqdm
from utils import *


def training_step(train_loader, model, optimizer, criterion, device):
    model.train()
    l1_loss = 0
    loop = tqdm(train_loader)
    for batch, (lr_hsi, hr_rgb, lr_rgb, hr_hsi) in enumerate(loop):
        lr_hsi, hr_rgb, lr_rgb, hr_hsi = lr_hsi.to(device), hr_rgb.to(device), lr_rgb.to(device), hr_hsi.to(device)
        y_pred = model(lr_rgb, hr_rgb, lr_hsi)
        loss = criterion(y_pred, hr_hsi)
        l1_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    l1_loss /= len(train_loader)
    return l1_loss


def validation_step(val_loader, model, criterion, device):
    model.eval()
    l1_loss = 0
    loop = tqdm(val_loader)
    with torch.inference_mode():
        for batch, (lr_hsi, hr_rgb, lr_rgb, hr_hsi) in enumerate(loop):
            lr_hsi, hr_rgb, lr_rgb, hr_hsi = lr_hsi.to(device), hr_rgb.to(device), lr_rgb.to(device), hr_hsi.to(device)
            y_pred = model(lr_rgb, hr_rgb, lr_hsi)
            loss = criterion(y_pred, hr_hsi)
            l1_loss += loss.item()
    l1_loss /= len(val_loader)
    return l1_loss


def train(train_loader, val_loader, model, criterion, optimizer, epochs, device, best_model_path, patience=30):
    early_stopping = EarlyStopping(patience=patience, mode='min')
    for epoch in tqdm(range(epochs)):
        train_loss = training_step(train_loader, model, optimizer, criterion, device)
        val_loss = validation_step(val_loader, model, criterion, device)
        print(
            f"Train loss: {train_loss:.4f} | "
            f"Validation loss: {val_loss:.4f} | "
        )
        print("-------------\n")
        if check_early_stopping(val_loss, model, early_stopping, epoch, best_model_path):
            break
    model.load_state_dict(torch.load(best_model_path))
    print(f"Restored best model weights with val_loss: {early_stopping.best_val:.4f}")
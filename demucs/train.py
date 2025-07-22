# train.py
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml
from datetime import datetime
from dataset import MUSDBDataset
from model import Demucs
from utils import save_checkpoint
from tqdm import tqdm
import pandas as pd
import time
from utils import EarlyStopping


# Load config
with open("demucs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Dataset & Dataloader
train_dataset = MUSDBDataset(split='train', config=config)
train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)

# Model
model = Demucs()
model.to(device)

# Loss & Optimizer
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

# Early stopping
early_stopper = EarlyStopping(patience=5, delta=1e-4)

# Training loop
loss_log = []
writer = SummaryWriter()
start = time.time()

for epoch in range(config["epochs"]):
    model.train()
    epoch_loss = 0
    for mix, targets in tqdm(train_loader):
        mix = mix.to(device)
        targets = {k: v.to(device) for k, v in targets.items()}

        # Predict
        estimates = model(mix)

        # Compute loss (sum over all instruments)
        loss = 0.0
        for k in config["target_instruments"]:
            pred = estimates[k]
            target = targets[k]

            # Trim target or prediction to the same length
            min_len = min(pred.shape[-1], target.shape[-1])
            pred = pred[..., :min_len]
            target = target[..., :min_len]

            loss += criterion(pred, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    epoch_loss = epoch_loss / len(train_loader)
    loss_log.append(epoch_loss)

    writer.add_scalar("Loss/train", epoch_loss, epoch)

    print(f"Epoch {epoch+1}/{config['epochs']}: Loss = {epoch_loss:.4f}")
    pd.DataFrame({"epoch": list(range(1, len(loss_log)+1)), "loss": loss_log}).to_csv("training_log.csv", index=False)

    # Check early stopping
    early_stopper(epoch_loss)
    if early_stopper.early_stop:
        print(f"Early stopping triggered at epoch {epoch+1}")
        break


print("Training time: {:.2f} mins".format((time.time() - start) / 60))
    
# Save model with real timestamp in the filename

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_path = os.path.join("checkpoints", f"demucs_epoch_{epoch+1}_{timestamp}.pth")
save_checkpoint(model, save_path)
    


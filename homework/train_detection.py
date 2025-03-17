import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils.tensorboard as tb
from homework.models import load_model, save_model
from homework.datasets.road_dataset import load_data

def train(
    exp_dir: str = "logs",
    model_name: str = "detector",
    num_epoch: int = 50,
    lr: float = 1e-3,
    batch_size: int = 128,
    seed: int = 2024,
    **kwargs,
):
    # Select device (GPU, MPS, or CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA running")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        print("Using MPS")
        device = torch.device("mps")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Logging setup
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    # Load model and move to device
    model = load_model(model_name, **kwargs).to(device)
    model.train()

    # Load dataset
    train_data = load_data("drive_data/train", transform_pipeline="default", shuffle=True, batch_size=batch_size, num_workers=4)
    val_data = load_data("drive_data/val", shuffle=False, batch_size=batch_size)

    # Loss functions
    optimizer = torch.optim.AdamW(model.parameters(), lr)
    criterion_seg = torch.nn.CrossEntropyLoss()
    criterion_depth = torch.nn.L1Loss()

    global_step = 0

    for epoch in range(num_epoch):
        model.train()
        train_losses = []

        for batch in train_data:
            img, depth, track = batch["image"].to(device), batch["depth"].to(device), batch["track"].to(device)
            
            logits, pred_depth = model(img)
            
            loss_seg = criterion_seg(logits, track)
            loss_depth = criterion_depth(pred_depth, depth.unsqueeze(1))
            loss = loss_seg + loss_depth
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            global_step += 1

        avg_train_loss = np.mean(train_losses)
        logger.add_scalar("train_loss", avg_train_loss, global_step)

        # Evaluation
        model.eval()
        val_losses = []

        with torch.no_grad():
            for batch in val_data:
                img, depth, track = batch["image"].to(device), batch["depth"].to(device), batch["track"].to(device)
                logits, pred_depth = model(img)
                
                loss_seg = criterion_seg(logits, track)
                loss_depth = criterion_depth(pred_depth, depth.unsqueeze(1))
                loss = loss_seg + loss_depth
                val_losses.append(loss.item())

        avg_val_loss = np.mean(val_losses)
        logger.add_scalar("val_loss", avg_val_loss, global_step)

        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
                f"train_loss={avg_train_loss:.4f} "
                f"val_loss={avg_val_loss:.4f}"
            )

    save_model(model)
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2024)

    train(**vars(parser.parse_args()))

"""
Author: Shreyas Dixit
This file contains the training loop for the Trajectory Prediction model.
"""
import os
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from BasicAI.functions.trainer.model import TrajectoryModel
from BasicAI.functions.dataset.dataloader import data_loader, save_scalers
from Config.config import load_config


def train(input_shape, output_shape, file_path, num_epochs, batch_size, weight_decay, lr, patience, checkpoint_dir):
    """
    Trains the Trajectory Prediction model with early stopping and model checkpointing.

    Args:
    - input_shape (int): The input shape of the model.
    - output_shape (int): The output shape of the model.
    - file_path (str): The path to the dataset file.
    - num_epochs (int): The number of epochs to train the model.
    - batch_size (int): The batch size for training.
    - weight_decay (float): The weight decay for the optimizer.
    - lr (float): The learning rate for the optimizer.
    - patience (int): Number of epochs to wait for improvement before early stopping.
    - checkpoint_dir (str): Directory to save model checkpoints.

    Returns:
    - None
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print("[Trainer] Initializing training...")

    if not Path(file_path).exists():
        raise FileNotFoundError(f"[Trainer] ERROR: Dataset file not found: {file_path}")

    print(f"[Trainer] Loading dataset from {file_path} with batch size {batch_size}")
    dataset_loader, input_scaler, output_scaler = data_loader(file_path, batch_size=batch_size, shuffle=True)

    print("[Trainer] Initializing model...")
    model = TrajectoryModel(input_shape, output_shape)

    # Initialize weights
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, verbose=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(f"[Trainer] Model initialized and moved to {device}")

    best_loss = float('inf')
    epochs_no_improve = 0
    best_model_path = checkpoint_dir / "best_model.pth"

    print(f"[Trainer] Starting training for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        print(f"[Trainer] Epoch {epoch + 1}/{num_epochs} in progress...")
        progress_bar = tqdm(dataset_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        epoch_loss = total_loss / len(dataset_loader)
        print(f"[Trainer] Epoch {epoch + 1}/{num_epochs} completed - Loss: {epoch_loss:.4f}")

        scheduler.step(epoch_loss)
        torch.cuda.empty_cache()

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            

            save_scalers(input_scaler, output_scaler, checkpoint_dir)
            print(f"[Trainer] New best model saved at {best_model_path}")
        else:
            epochs_no_improve += 1
            print(f"[Trainer] No improvement for {epochs_no_improve}/{patience} epochs")

        # Early stopping with model restoration
        if epochs_no_improve >= patience:
            print("[Trainer] Early stopping triggered. Restoring best model...")
            model.load_state_dict(torch.load(best_model_path))
            break

    print("[Trainer] Training completed!")


# if __name__ == "__main__":
#     print("[Trainer] Loading configuration file...")
#     config = load_config("Config/basic_ai_config.yaml")['Architecture']['NeuralNet']

#     print("[Trainer] Starting model training...")
#     train(
#         input_shape=config['input_shape'], 
#         output_shape=config['output_shape'], 
#         file_path=config['dataset'], 
#         num_epochs=config['num_epochs'], 
#         batch_size=config['batch_size'], 
#         weight_decay=config['weight_decay'], 
#         lr=config['lr'], 
#         patience=config['patience'], 
#         checkpoint_dir=config['checkpoint_dir']
#     )
#     print("[Trainer] Training process finished!")

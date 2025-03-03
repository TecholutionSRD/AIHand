"""
Author: Shreyas Dixit
A PyTorch Dataset class for the Trajectory Prediction Problem.
"""
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import joblib

class TrajectoryDataset(Dataset):
    """
    A PyTorch Dataset class for loading and preprocessing trajectory data.
    Args:
        file_path (str): Path to the CSV file containing the dataset.
        scaler_dir (str, optional): Directory path to load existing scalers. If not provided, new scalers will be created and fitted.
    Attributes:
        data (pd.DataFrame): The loaded dataset.
        inputs (np.ndarray): The input features from the dataset.
        outputs (np.ndarray): The output targets from the dataset.
        input_scaler (StandardScaler): Scaler for normalizing input features.
        output_scaler (StandardScaler): Scaler for normalizing output targets.
    Methods:
        __len__(): Returns the number of samples in the dataset.
        __getitem__(idx): Returns the input and output tensors for the given index.
    """
    def __init__(self, file_path, scaler_dir=None):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        self.data = pd.read_csv(file_path)
        self.inputs = self.data.iloc[:, :6].values.astype('float32')
        self.outputs = self.data.iloc[:, 6:].values.astype('float32')

        self.input_scaler = StandardScaler()
        self.output_scaler = StandardScaler()

        if scaler_dir and Path(scaler_dir).exists():
            self.input_scaler = joblib.load(Path(scaler_dir) / 'input_scaler.pkl')
            self.output_scaler = joblib.load(Path(scaler_dir) / 'output_scaler.pkl')
        else:
            self.inputs = self.input_scaler.fit_transform(self.inputs)
            self.outputs = self.output_scaler.fit_transform(self.outputs)

    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.inputs[idx], dtype=torch.float32)
        y = torch.tensor(self.outputs[idx], dtype=torch.float32)
        return x, y

def data_loader(file_path, batch_size=8, shuffle=True, device="cpu", scaler_dir=None):
    """
    Create DataLoader for the dataset and return scalers.
    """
    dataset = TrajectoryDataset(file_path, scaler_dir=scaler_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader, dataset.input_scaler, dataset.output_scaler

def save_scalers(input_scaler, output_scaler, checkpoint_dir):
    """
    Save input and output scalers to the checkpoint directory.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(input_scaler, checkpoint_dir / 'input_scaler.pkl')
    joblib.dump(output_scaler, checkpoint_dir / 'output_scaler.pkl')
    print(f"Scalers saved to {checkpoint_dir}")

def load_scalers(checkpoint_dir):
    """
    Load input and output scalers from the checkpoint directory.
    """
    checkpoint_dir = Path(checkpoint_dir)
    input_scaler = joblib.load(checkpoint_dir / 'input_scaler.pkl')
    output_scaler = joblib.load(checkpoint_dir / 'output_scaler.pkl')
    print(f"Scalers loaded from {checkpoint_dir}")
    return input_scaler, output_scaler

if __name__ == "__main__":
    file_path = 'dataset.csv'
    
    try:
        dataloader, input_scaler, output_scaler = data_loader(file_path, scaler_dir="scalers/")
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            print(f"Batch {batch_idx + 1}:")
            print(f"Inputs: {inputs.shape}")
            print(f"Targets: {targets.shape}")
            break
    except FileNotFoundError as e:
        print(f"Error: {e}")

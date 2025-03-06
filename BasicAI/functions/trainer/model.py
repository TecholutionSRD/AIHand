"""
Author : Shreyas Dixit
This file contains a modified neural network model for the Trajectory Prediction Problem,
optimized for a small dataset of 60 samples.
"""
import torch
import torch.nn as nn
from Config.config import load_config

class TrajectoryModel(nn.Module):
    """
    A neural network model for the Trajectory Prediction Problem, designed for small datasets.
        
    Args:
        input_shape (int): The number of input features (default=6).
        output_shape (int): The number of output features (default=64).
        dropout_rate (float): Dropout probability to prevent overfitting (default=0.1).

    Returns:
        torch.Tensor: The output tensor of shape (batch_size, output_shape).
    """
    def __init__(self, input_shape=6, output_shape=64, dropout_rate=0.1):
        super(TrajectoryModel, self).__init__()
        
        self.fc1 = nn.Linear(input_shape, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(64, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(128, output_shape)
        self.bn3 = nn.BatchNorm1d(output_shape)

        # Weight Initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.drop1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.drop2(x)
        
        x = self.fc3(x)
        x = self.bn3(x) 
        
        return x
    
#-------------------------------------------------------------------------------------------#
def build_model(config_path: str, input_shape:int, output:int):
    config = load_config(config_path)['Architecture']['NeuralNet']
    print("[Architecture] Model Architecture")
    print("-" * 100)
    print(f"[Training] Input :  {input_shape}")
    print(f"[Training] Output :  {output}")
    print(f"[Training] Dropout: {config.get('dropout', 0.1)}")
    print(f"[Training] Number of Epochs: {config.get('num_epochs', 100)}")
    print(f"[Training] Batch Size: {config.get('batch_size', 8)}")
    print(f"[Training] Weight Decay: {config.get('weight_decay', 0.00001)}")
    print(f"[Training] Learning Rate: {config.get('lr', 0.001)}")
    print(f"[Training] Patience: {config.get('patience', 10)}")
    model = TrajectoryModel(input_shape, output, config.get('dropout', 0.1))
    print("[Model] Model Built")
    return model

if __name__ == "__main__":
    model = TrajectoryModel()
    x = torch.randn(10, 6)
    y = model(x)
    print("Output shape:", y.shape) 

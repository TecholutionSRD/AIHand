"""
Author: Shreyas Dixit
This file contains the inference class for the base model.
"""
from pathlib import Path
import joblib
import torch
import os
import sys

# Ensure proper imports from the project structure
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from BasicAI.functions.trainer.model import TrajectoryModel
from Config.config import load_config


class TrajectoryInference:
    """Class for performing inference with the trained Trajectory Prediction model."""

    def __init__(self, config_path: str = "basic_ai_config.yaml", checkpoint_dir: str = "checkpoints"):
        """
        Initialize the inference class by loading model, scalers, and configurations.
        
        Args:
            config_path (str): Path to the YAML configuration file.
            checkpoint_dir (str): Directory containing model checkpoints and scalers.
        """
        print("[Inference] Initializing inference module...")

        config = load_config(config_path)['Architecture']['NeuralNet']
        self.input_shape = config['input_shape']
        self.output_shape = config['output_shape']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.checkpoint_dir = Path(checkpoint_dir)
        self.model_path = self.checkpoint_dir / "best_model.pth"
        self.input_scaler_path = self.checkpoint_dir / "input_scaler.pkl"
        self.output_scaler_path = self.checkpoint_dir / "output_scaler.pkl"

        self._check_files_exist()

        self.model = self._load_model()
        self.input_scaler, self.output_scaler = self._load_scalers()

    def _check_files_exist(self):
        """Ensure required files (model & scalers) exist before proceeding."""
        missing_files = [f for f in [self.model_path, self.input_scaler_path, self.output_scaler_path] if not f.exists()]
        if missing_files:
            raise FileNotFoundError(f"[Inference] ERROR: Missing required files: {missing_files}")

    def _load_model(self) -> TrajectoryModel:
        """Load the trained model from checkpoint."""
        print(f"[Inference] Loading model from {self.model_path}...")
        model = TrajectoryModel(self.input_shape, self.output_shape).to(self.device)
        try:
            model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            model.eval()
            print("[Inference] Model loaded successfully.")
        except RuntimeError as e:
            print(f"[Inference] Model weights do not match. Please check the configs {e}")
            raise
        return model

    def _load_scalers(self):
        """Load input and output scalers."""
        print("[Inference] Loading scalers...")
        input_scaler = joblib.load(self.input_scaler_path)
        output_scaler = joblib.load(self.output_scaler_path)
        print("[Inference] Scalers loaded successfully.")
        return input_scaler, output_scaler

    def predict(self, points: list) -> list:
        """
        Perform inference using the trained model.

        Args:
            points (list): List of two 3D coordinates (x1, y1, z1, x2, y2, z2).

        Returns:
            list: Predicted trajectory points in (X, Y, Z, C) format.
        """
        print("[Inference] Preprocessing input data...")

        points_scaled = self.input_scaler.transform([sum(points, [])])
        points_tensor = torch.tensor(points_scaled, dtype=torch.float32).to(self.device)

        print("[Inference] Running model prediction...")
        with torch.no_grad():
            output = self.model(points_tensor)
            output = output.cpu().numpy()

        print("[Inference] Transforming output back to original scale...")
        output = self.output_scaler.inverse_transform(output).reshape(-1, 4).tolist()

        return output

    def save_output(self, output: list, output_path: Path = Path("output/output.csv")):
        """
        Save the predicted output to a CSV file.

        Args:
            output (list): List of predicted trajectory points.
            output_path (Path): Path where the output CSV should be saved.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open('w') as f:
            f.write('X,Y,Z,C\n')
            for line in output:
                f.write(','.join(map(str, line)) + '\n')

    def run_inference(self, points: list) -> Path:
        """
        Perform inference and save the output.

        Args:
            points (list): List of 6 values representing two points.

        Returns:
            Path: Path to the output CSV file.
        """
        print("[Inference] Starting inference process...")
        predictions = self.predict(points)
        output_path = Path("output/output.csv")
        self.save_output(predictions, output_path)
        print("[Inference] Process complete.")
        return output_path


# if __name__ == '__main__':
#     print("[Inference] Running script...")

#     # Sample test input points
#     test_points = [[-370.808101, -735.661648, 104.0947533], 
#                    [-202.9524806, -828.6285971, 41.35059464]]

#     # Create inference instance and run prediction
#     inference_engine = TrajectoryInference()
#     csv_path = inference_engine.run_inference(test_points)

#     print(f"[Inference] Output saved at {csv_path}")

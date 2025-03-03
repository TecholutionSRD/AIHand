# BasicAI

**BasicAI** is an end-to-end pipeline designed to process videos, extract trajectories, train models, and perform inference. It enables users to analyze object movements, predict future positions, and automate trajectory-based decision-making.

## 🚀 Features

- **PreProcessing Pipeline**: Cleans up dataset folders, renames them, and generates a merged CSV.
- **Video Processing**: Extracts object trajectories (4D) from video input.
- **Model Training**: Neural network training on extracted trajectories.
- **Inference System**: Predicts future movements based on trained models.
- **Efficient Data Handling**: Automated scaling, transformation, and storage.
- **REST API Support**: Expose endpoints for easy integration.

---

## 📂 Project Structure

```
BasicAI/
│── PreProcessor/
│   ├── preprocessor.py  # Cleans and prepares dataset for training
│── VideoProcessor/
│   ├── video_processor.py  # Extracts trajectories from video
│── Training/
│   ├── train.py  # Model training script
│── Inference/
│   ├── inference.py  # Model inference script
│── Config/
│   ├── config.yaml  # Configuration settings
│── API/
│   ├── app.py  # FastAPI application with endpoints
│── README.md  # Project documentation
│── requirements.txt  # Dependencies
```

---

## 🛠️ Setup & Installation

### Prerequisites

Ensure you have the following dependencies installed:

- Python 3.7+
- PyTorch
- scikit-learn
- NumPy
- OpenCV (for video processing)
- FastAPI (for API)
- Uvicorn (for serving API)

### Installation

Clone the repository and install dependencies:
```bash
git clone https://github.com/Techolution/AIHand.git
cd AIHand
pip install -r requirements.txt
```

Update the `config/config.yaml` file to match your setup.

---

## 🔄 PreProcessing Pipeline

Before training, we need to ensure the dataset is structured correctly.

### **PreProcessor Workflow**
1. **Check if `merged_csv.csv` exists** → Skip processing if found.
2. **Filter out invalid folders** → Remove subfolders without `processed_predictions_hamer.csv`.
3. **Rename folders sequentially** → `sample_1`, `sample_2`, ... `sample_n`.
4. **Generate Merged CSV** → Calls `VideoProcessor4D` to create `merged_csv.csv`.

### **Run PreProcessor**
```bash
python PreProcessor/preprocessor.py --action_name "walking"
```

---

## 📊 Training the Model

### **Train the Model**
```bash
python Training/train.py
```
This will:
- Load the dataset
- Train the model with `num_epochs`, `batch_size`, and `lr` from `config.yaml`
- Save the best model checkpoint to `checkpoints/best_model.pth`

---

## 🎯 Inference

### **Run Model Prediction**
```bash
python Inference/inference.py
```
This will:
- Load the trained model
- Process new trajectory data
- Output predictions to a CSV file

### **Example Input**
```python
from Inference.inference import InferenceProcessor

points = [[-370.808101,-735.661648,104.0947533], [-202.9524806,-828.6285971,41.35059464]]
inference_processor = InferenceProcessor("checkpoints/best_model.pth")
csv_path = inference_processor.run_inference(points)

print(f"Predictions saved at {csv_path}")
```
**Output:** `output/output.csv`

---

## 📡 API Endpoints

### **1️⃣ Upload Video for Processing**
```http
POST /upload
```
**Description:** Uploads a video and processes it to extract object trajectories.

**Payload:**  
```json
{
  "video_path": "path/to/video.mp4"
}
```
**Response:**  
```json
{
  "message": "Video processed successfully",
  "output_csv": "output/processed_data.csv"
}
```

---

### **2️⃣ Train Model**
```http
POST /train
```
**Description:** Triggers model training on the preprocessed dataset.

**Response:**  
```json
{
  "message": "Training started",
  "checkpoint_path": "checkpoints/best_model.pth"
}
```

---

### **3️⃣ Run Inference**
```http
POST /predict
```
**Description:** Runs inference using a trained model.

**Payload:**  
```json
{
  "points": [[-370.808101,-735.661648,104.0947533], [-202.9524806,-828.6285971,41.35059464]]
}
```
**Response:**  
```json
{
  "message": "Inference complete",
  "output_csv": "output/predictions.csv"
}
```

---

## 🤝 Contributing

We welcome contributions! To contribute:
1. Fork the repository.
2. Create a new branch (`feature-new`).
3. Make changes and submit a PR.

---

## 📜 License

This project is licensed under the MIT License. See `LICENSE` for details.



from ultralytics import YOLO
from pathlib import Path

# Teacher Eye - Student Engagement Model Training

def train_model():
    model_path = Path("yolov9c.pt")
    data_path = Path("predictions/Student-Attention-Tracking-2/data.yaml")

    if not data_path.exists():
        print("Error: Dataset data.yaml file not found.")
        print("Check the dataset folder and data.yaml location.")
        return

    print("Loading YOLO model...")
    model = YOLO(str(model_path))

    print("Starting student engagement model training...")

    model.train(
        data=str(data_path),
        epochs=30,
        imgsz=640,
        batch=8,
        project="runs/teacher_eye",
        name="student_engagement"
    )

    print("Training completed!")

if __name__ == "__main__":
    train_model()

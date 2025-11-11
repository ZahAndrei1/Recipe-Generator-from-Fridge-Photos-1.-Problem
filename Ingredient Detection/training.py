from ultralytics import YOLO
import torch

# Check GPU availability
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

if __name__ == '__main__':
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

    model = YOLO('yolov8n.pt')

    model.train(
        data='ingredients_data.yaml',
        epochs=20,
        imgsz=640,
        batch=16,
        device=0,
        patience=10,
        project='runs/ingredient_detection',
        name='demo'
    )
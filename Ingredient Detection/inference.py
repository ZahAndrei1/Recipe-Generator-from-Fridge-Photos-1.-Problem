from ultralytics import YOLO
from PIL import Image

# Load your trained model
model = YOLO('runs/ingredient_detection/demo3/weights/best.pt')

# Run prediction
results = model('test_image.jpg')

# Display results
for result in results:
    result.show()  # Shows image with bounding boxes
    
    # Print detected ingredients
    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        ingredient = model.names[class_id]
        print(f"Detected: {ingredient} ({confidence:.2f})")
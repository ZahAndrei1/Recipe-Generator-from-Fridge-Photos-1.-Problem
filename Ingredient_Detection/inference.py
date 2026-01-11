from ultralytics import YOLO
from PIL import Image
import os

# IMPORTANT: This must match the model's training classes (from ingredients_data.yaml)
INGREDIENT_CLASSES = [
    'Apple', 'Banana', 'Orange', 'Tomato', 'Carrot',
    'Potato', 'Bread', 'Cheese', 'Broccoli', 'Strawberry',
    'Lemon', 'Cucumber', 'Onion', 'Garlic', 'Mushroom',
    'Lettuce', 'Egg', 'Chicken', 'Fish', 'Shrimp',
    'Milk', 'Butter', 'Rice', 'Pasta', 'Corn'
]

# Load your trained model (use absolute path relative to this file)
_current_dir = os.path.dirname(os.path.abspath(__file__))
_model_path = os.path.join(_current_dir, 'runs', 'ingredient_detection', 'demo5', 'weights', 'best.pt')
model = YOLO(_model_path)

def detect_ingredients(image_path: str):
    """
    Runs YOLO on the given image and returns a list of unique detected ingredient names.
    """
    results = model(image_path)
    detected = set()

    for r in results:
        boxes = r.boxes
        if boxes is None:
            continue
        for box in boxes:
            cls_id = int(box.cls[0])
            name = model.names[cls_id]
            # Optional: map YOLO class name → your INGREDIENT_CLASSES name
            detected.add(name)

    return list(detected)

if __name__ == "__main__":
    # Run inference on test image
    test_image = os.path.join(_current_dir, 'Fridge_Photo.jpeg')

    print(f"Running inference on: {test_image}\n")

    # Get detection results
    results = model(test_image)

    # Display results with bounding boxes
    for result in results:
        result.show()  # Shows image with bounding boxes

        # Print detected ingredients with confidence scores
        print("Detected ingredients:")
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            ingredient = model.names[class_id]
            print(f"  - {ingredient}: {confidence:.2%} confidence")

    # Also print the list of unique ingredients
    print(f"\nUnique ingredients: {detect_ingredients(test_image)}")
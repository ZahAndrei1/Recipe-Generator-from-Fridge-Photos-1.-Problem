import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")

    # Create the YAML file
yaml_content = """train: ingredients_dataset/train/images
val: ingredients_dataset/train/images

nc: 10
names: ['Apple', 'Banana', 'Orange', 'Tomato', 'Carrot', 'Potato', 'Bread', 'Cheese', 'Broccoli', 'Strawberry']
"""

with open('ingredients_data.yaml', 'w') as f:
    f.write(yaml_content)

print("Created ingredients_data.yaml")
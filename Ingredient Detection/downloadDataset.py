import requests
import pandas as pd
from pathlib import Path
import urllib.request
from tqdm import tqdm
import os

# Download class descriptions
def download_class_descriptions():
    url = "https://storage.googleapis.com/openimages/v6/oidv6-class-descriptions.csv"
    df = pd.read_csv(url, names=['code', 'name'])
    return df

# Find class codes
def get_ingredient_codes():
    df = download_class_descriptions()
    
    ingredients = ['Apple', 'Banana', 'Orange', 'Tomato', 'Carrot', 
                   'Potato', 'Bread', 'Cheese', 'Broccoli', 'Strawberry']
    
    codes = {}
    for ingredient in ingredients:
        match = df[df['name'] == ingredient]
        if not match.empty:
            codes[ingredient] = match.iloc[0]['code']
            print(f"Found: {ingredient} -> {match.iloc[0]['code']}")
    
    return codes

# Download images and annotations
def download_ingredients_dataset(limit_per_class=500):
    ingredient_codes = get_ingredient_codes()
    
    # Download annotations
    train_url = "https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv"
    print("Downloading annotations...")
    annotations = pd.read_csv(train_url)
    
    # Filter for ingredients
    annotations = annotations[annotations['LabelName'].isin(ingredient_codes.values())]
    
    # Create directories
    Path('ingredients_dataset/train/images').mkdir(parents=True, exist_ok=True)
    Path('ingredients_dataset/train/labels').mkdir(parents=True, exist_ok=True)
    
    # Get image IDs
    image_ids = annotations['ImageID'].unique()[:limit_per_class]
    
    print(f"Downloading {len(image_ids)} images...")
    
    # Download image list first
    print("Downloading image URLs...")
    images_url = "https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable-with-rotation.csv"
    images_df = pd.read_csv(images_url)
    
    success_count = 0
    for img_id in tqdm(image_ids):
        # Get the actual image URL
        img_row = images_df[images_df['ImageID'] == img_id]
        if img_row.empty:
            continue
            
        img_url = img_row.iloc[0]['OriginalURL']
        img_path = f'ingredients_dataset/train/images/{img_id}.jpg'
        
        try:
            urllib.request.urlretrieve(img_url, img_path)
            
            # Verify image was downloaded
            if os.path.exists(img_path) and os.path.getsize(img_path) > 0:
                success_count += 1
                
                # Create YOLO label
                img_annotations = annotations[annotations['ImageID'] == img_id]
                with open(f'ingredients_dataset/train/labels/{img_id}.txt', 'w') as f:
                    for _, row in img_annotations.iterrows():
                        class_id = list(ingredient_codes.values()).index(row['LabelName'])
                        x_center = (row['XMin'] + row['XMax']) / 2
                        y_center = (row['YMin'] + row['YMax']) / 2
                        width = row['XMax'] - row['XMin']
                        height = row['YMax'] - row['YMin']
                        f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
        except Exception as e:
            continue
    
    print(f"Successfully downloaded {success_count} images")
    
    # Create data.yaml
    yaml_content = f"""train: ingredients_dataset/train/images
val: ingredients_dataset/train/images

nc: {len(ingredient_codes)}
names: {list(ingredient_codes.keys())}
"""
    with open('ingredients_data.yaml', 'w') as f:
        f.write(yaml_content)
    
    print("Dataset ready!")

# Run
download_ingredients_dataset(limit_per_class=500)
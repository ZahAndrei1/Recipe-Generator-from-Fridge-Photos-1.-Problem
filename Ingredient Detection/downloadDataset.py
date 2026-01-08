import requests
import pandas as pd
from pathlib import Path
import urllib.request
from tqdm import tqdm
import os
import socket

# *** SET GLOBAL TIMEOUT: 10 seconds to prevent hanging ***
socket.setdefaulttimeout(10)

# Download class descriptions
def download_class_descriptions():
    url = "https://storage.googleapis.com/openimages/v6/oidv6-class-descriptions.csv"
    df = pd.read_csv(url, names=['code', 'name'])
    return df

# Find class codes
def get_ingredient_codes():
    df = download_class_descriptions()
    
    # *** EXPANDED INGREDIENT LIST ***
    ingredients = ['Apple', 'Banana', 'Orange', 'Tomato', 'Carrot', 
                   'Potato', 'Bread', 'Cheese', 'Broccoli', 'Strawberry',
                   'Lemon', 'Cucumber', 'Onion', 'Garlic', 'Mushroom',
                   'Lettuce', 'Egg', 'Chicken', 'Fish', 'Shrimp',
                   'Milk', 'Butter', 'Rice', 'Pasta', 'Corn']
    # *** NOW 25 INGREDIENTS INSTEAD OF 10 ***
    
    codes = {}
    for ingredient in ingredients:
        match = df[df['name'] == ingredient]
        if not match.empty:
            codes[ingredient] = match.iloc[0]['code']
            print(f"Found: {ingredient} -> {match.iloc[0]['code']}")
    
    return codes

# Download images and annotations
def download_ingredients_dataset(limit_per_class=300):
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
    
    # *** MODIFIED: Get image IDs per class to ensure limit_per_class images for each ingredient ***
    image_ids_set = set()
    for ingredient_code in ingredient_codes.values():
        class_annotations = annotations[annotations['LabelName'] == ingredient_code]
        class_image_ids = class_annotations['ImageID'].unique()[:limit_per_class]
        image_ids_set.update(class_image_ids)
    
    image_ids = list(image_ids_set)
    print(f"Total unique images to download: {len(image_ids)}")
    # *** END MODIFICATION ***
    
    print(f"Downloading {len(image_ids)} images...")
    
    success_count = 0
    failed_count = 0
    
    # *** OPTIMIZED: Construct image URLs directly instead of downloading giant CSV ***
    # OpenImages uses a predictable URL pattern
    base_url = "https://s3.amazonaws.com/open-images-dataset"
    
    for img_id in tqdm(image_ids):
        # Construct the image URL directly (OpenImages URL pattern)
        img_url = f"{base_url}/train/{img_id}.jpg"
        img_path = f'ingredients_dataset/train/images/{img_id}.jpg'
        
        try:
            # *** TIMEOUT SET GLOBALLY via socket.setdefaulttimeout(10) ***
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
            else:
                # If S3 fails, remove the bad file
                if os.path.exists(img_path):
                    os.remove(img_path)
                failed_count += 1
        except Exception as e:
            # Clean up failed downloads
            if os.path.exists(img_path):
                os.remove(img_path)
            failed_count += 1
            continue
    
    print(f"\n{'='*50}")
    print(f"Successfully downloaded {success_count} images")
    print(f"Failed downloads: {failed_count}")
    print(f"Success rate: {success_count}/{len(image_ids)} ({100*success_count/len(image_ids):.1f}%)")
    print(f"{'='*50}\n")
    
    # Create data.yaml
    yaml_content = f"""train: ingredients_dataset/train/images
val: ingredients_dataset/train/images

nc: {len(ingredient_codes)}
names: {list(ingredient_codes.keys())}
"""
    with open('ingredients_data.yaml', 'w') as f:
        f.write(yaml_content)
    
    print("Dataset ready!")

# *** CHANGED: Download 300 images per ingredient class ***
download_ingredients_dataset(limit_per_class=300)
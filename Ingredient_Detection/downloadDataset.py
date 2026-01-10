import requests
import pandas as pd
from pathlib import Path
import urllib.request
from tqdm import tqdm
import os
import socket
import time
import shutil

# *** SET GLOBAL TIMEOUT: 15 seconds ***
socket.setdefaulttimeout(15)

def download_class_descriptions():
    url = "https://storage.googleapis.com/openimages/v6/oidv6-class-descriptions.csv"
    df = pd.read_csv(url, names=['code', 'name'])
    return df

def get_ingredient_codes():
    df = download_class_descriptions()
    
    # Use the exact codes from the diagnostic
    ingredient_codes = {
        'Apple': '/m/014j1m',
        'Banana': '/m/09qck',
        'Orange': '/m/0cyhj_',
        'Tomato': '/m/07j87',
        'Carrot': '/m/0fj52s',
        'Potato': '/m/05vtc',
        'Bread': '/m/09728',
        'Cheese': '/m/01nkt',
        'Broccoli': '/m/0hkxq',
        'Strawberry': '/m/07fbm7',
        'Lemon': '/m/09k_b',
        'Cucumber': '/m/015x4r',
        'Onion': '/m/0dj75',
        'Garlic': '/m/0dbrl',
        'Mushroom': '/m/052sf',
        'Lettuce': '/m/0fqlj',
        'Egg': '/m/02g387',
        'Chicken': '/m/09b5t',
        'Fish': '/m/0ch_cf',
        'Shrimp': '/m/0ll1f78',
        'Milk': '/m/04zpv',
        'Butter': '/m/0chyh',
        'Rice': '/m/09759',
        'Pasta': '/m/05z55',
        'Corn': '/m/053wn'
    }
    
    print("Using verified ingredient codes:")
    for name, code in ingredient_codes.items():
        print(f"  {name:20s} -> {code}")
    
    return ingredient_codes

def download_ingredients_dataset(limit_per_class=300, max_retries=3, clean_start=True):
    ingredient_codes = get_ingredient_codes()
    
    # Clean existing data if requested
    if clean_start:
        dataset_path = Path('ingredients_dataset')
        if dataset_path.exists():
            print("\n⚠️  Removing existing dataset...")
            shutil.rmtree(dataset_path)
            print("✓ Old data removed\n")
    
    # Download annotations
    train_url = "https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv"
    print("Downloading annotations (this may take a while)...")
    annotations = pd.read_csv(train_url)
    print(f"Total annotations in dataset: {len(annotations)}")
    
    # Filter for ingredients
    annotations = annotations[annotations['LabelName'].isin(ingredient_codes.values())]
    print(f"Ingredient annotations found: {len(annotations)}")
    
    # Check available annotations per category
    print("\n" + "="*60)
    print("ANNOTATIONS AVAILABLE PER CATEGORY:")
    print("="*60)
    for ingredient_name, ingredient_code in ingredient_codes.items():
        count = len(annotations[annotations['LabelName'] == ingredient_code])
        print(f"{ingredient_name:20s}: {count:5d} annotations available")
    print("="*60 + "\n")
    
    # Create directories
    Path('ingredients_dataset/train/images').mkdir(parents=True, exist_ok=True)
    Path('ingredients_dataset/train/labels').mkdir(parents=True, exist_ok=True)
    
    # Get balanced image IDs per class
    image_ids_set = set()
    
    for ingredient_name, ingredient_code in ingredient_codes.items():
        class_annotations = annotations[annotations['LabelName'] == ingredient_code]
        class_image_ids = class_annotations['ImageID'].unique()[:limit_per_class]
        image_ids_set.update(class_image_ids)
        print(f"Selected {len(class_image_ids)} images for {ingredient_name}")
    
    image_ids = list(image_ids_set)
    print(f"\nTotal unique images to download: {len(image_ids)}")
    
    # Download images
    success_count = 0
    failed_count = 0
    base_url = "https://s3.amazonaws.com/open-images-dataset"
    
    for img_id in tqdm(image_ids, desc="Downloading images"):
        img_url = f"{base_url}/train/{img_id}.jpg"
        img_path = f'ingredients_dataset/train/images/{img_id}.jpg'
        
        # Skip if already downloaded
        if os.path.exists(img_path) and os.path.getsize(img_path) > 0:
            success_count += 1
            continue
        
        # Retry logic
        downloaded = False
        for attempt in range(max_retries):
            try:
                urllib.request.urlretrieve(img_url, img_path)
                
                if os.path.exists(img_path) and os.path.getsize(img_path) > 0:
                    success_count += 1
                    downloaded = True
                    
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
                    break
                else:
                    if os.path.exists(img_path):
                        os.remove(img_path)
                    if attempt < max_retries - 1:
                        time.sleep(1)  # Wait before retry
            except Exception as e:
                if os.path.exists(img_path):
                    os.remove(img_path)
                if attempt < max_retries - 1:
                    time.sleep(1)
                continue
        
        if not downloaded:
            failed_count += 1
    
    print(f"\n{'='*60}")
    print(f"Successfully downloaded: {success_count} images")
    print(f"Failed downloads: {failed_count}")
    print(f"Success rate: {success_count}/{len(image_ids)} ({100*success_count/len(image_ids):.1f}%)")
    print(f"{'='*60}\n")
    
    # Create data.yaml
    yaml_content = f"""train: ingredients_dataset/train/images
val: ingredients_dataset/train/images

nc: {len(ingredient_codes)}
names: {list(ingredient_codes.keys())}
"""
    with open('ingredients_data.yaml', 'w') as f:
        f.write(yaml_content)
    
    print("✓ Dataset ready!")
    print("✓ Run your distribution check script to verify the data")

# Download with clean start
download_ingredients_dataset(limit_per_class=300, max_retries=3, clean_start=True)
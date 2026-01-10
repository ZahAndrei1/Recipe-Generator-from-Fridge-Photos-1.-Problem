import pandas as pd
from pathlib import Path
import urllib.request
from tqdm import tqdm
import os
import socket
import time
from collections import Counter
import tarfile
import shutil

socket.setdefaulttimeout(15)

def check_current_distribution(labels_dir):
    """Check which categories need more data"""
    
    labels_path = Path(labels_dir)
    category_counts = Counter()
    
    for label_file in labels_path.glob('*.txt'):
        with open(label_file, 'r') as f:
            for line in f:
                if line.strip():
                    class_id = int(line.split()[0])
                    category_counts[class_id] += 1
    
    ingredient_names = ['Apple', 'Banana', 'Orange', 'Tomato', 'Carrot', 
                       'Potato', 'Bread', 'Cheese', 'Broccoli', 'Strawberry',
                       'Lemon', 'Cucumber', 'Onion', 'Garlic', 'Mushroom',
                       'Lettuce', 'Egg', 'Chicken', 'Fish', 'Shrimp',
                       'Milk', 'Butter', 'Rice', 'Pasta', 'Corn']
    
    return category_counts, ingredient_names

def get_categories_needing_data(labels_dir, min_threshold=500):
    """Identify which categories need supplementing"""
    
    category_counts, ingredient_names = check_current_distribution(labels_dir)
    
    needs_supplement = {}
    
    print("\n" + "="*70)
    print("CATEGORIES NEEDING SUPPLEMENTATION")
    print("="*70)
    
    for class_id, name in enumerate(ingredient_names):
        count = category_counts.get(class_id, 0)
        if count < min_threshold:
            needed = min_threshold - count
            needs_supplement[name] = {
                'class_id': class_id,
                'current_count': count,
                'needed': needed
            }
            print(f"✗ {name:20s}: {count:4d} current, need {needed:4d} more")
        else:
            print(f"✓ {name:20s}: {count:4d} (sufficient)")
    
    print("="*70 + "\n")
    
    return needs_supplement, ingredient_names

def download_food101_dataset():
    """Download Food-101 dataset"""
    
    food101_url = "http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"
    food101_path = "food-101.tar.gz"
    
    print("Downloading Food-101 dataset (this is ~5GB, may take a while)...")
    
    if not os.path.exists(food101_path):
        try:
            urllib.request.urlretrieve(food101_url, food101_path)
            print("✓ Download complete!")
        except Exception as e:
            print(f"❌ Error downloading Food-101: {e}")
            return None
    else:
        print("✓ Food-101 already downloaded")
    
    # Extract dataset
    extract_dir = "food-101"
    if not os.path.exists(extract_dir):
        print("Extracting Food-101 dataset...")
        try:
            with tarfile.open(food101_path, 'r:gz') as tar:
                tar.extractall()
            print("✓ Extraction complete!")
        except Exception as e:
            print(f"❌ Error extracting: {e}")
            return None
    else:
        print("✓ Food-101 already extracted")
    
    return extract_dir

def map_categories_to_food101():
    """Map our categories to Food-101 categories"""
    
    # Food-101 has 101 food categories
    # Map our ingredients to Food-101 categories
    category_mapping = {
        'Apple': 'apple_pie',  # Closest match in Food-101
        'Banana': None,  # Not directly in Food-101
        'Orange': None,  # Not directly in Food-101
        'Tomato': None,  # Not directly in Food-101
        'Carrot': 'carrot_cake',  # Closest match
        'Potato': 'french_fries',  # Contains potatoes
        'Bread': 'bread_pudding',  # Contains bread
        'Cheese': 'cheese_plate',  # If available
        'Broccoli': None,
        'Strawberry': 'strawberry_shortcake',  # Closest match
        'Lemon': None,
        'Cucumber': None,
        'Onion': 'onion_rings',  # Contains onions
        'Garlic': 'garlic_bread',  # If available
        'Mushroom': None,
        'Lettuce': 'caesar_salad',  # Contains lettuce
        'Egg': 'eggs_benedict',  # Contains eggs
        'Chicken': 'chicken_wings',  # Multiple chicken dishes available
        'Fish': 'fish_and_chips',  # Contains fish
        'Shrimp': 'shrimp_and_grits',  # If available
        'Milk': None,
        'Butter': None,
        'Rice': 'fried_rice',  # Contains rice
        'Pasta': 'spaghetti_carbonara',  # Contains pasta
        'Corn': None
    }
    
    return category_mapping

def supplement_from_food101(needs_supplement, images_dir, labels_dir):
    """Supplement dataset using Food-101"""
    
    # Download Food-101
    food101_dir = download_food101_dataset()
    
    if not food101_dir:
        print("❌ Could not download Food-101")
        return 0
    
    # Get category mapping
    category_mapping = map_categories_to_food101()
    
    total_added = 0
    food101_images_path = Path(food101_dir) / "food-101" / "images"
    
    for category, info in needs_supplement.items():
        food101_category = category_mapping.get(category)
        
        if not food101_category:
            print(f"⊘ {category}: Not available in Food-101, skipping")
            continue
        
        category_path = food101_images_path / food101_category
        
        if not category_path.exists():
            print(f"⊘ {category} -> {food101_category}: Directory not found, skipping")
            continue
        
        # Get images from Food-101 category
        source_images = list(category_path.glob('*.jpg'))
        
        if not source_images:
            print(f"⊘ {category}: No images found in {food101_category}")
            continue
        
        # Limit to what we need
        num_to_copy = min(info['needed'], len(source_images))
        
        print(f"\n{category} -> {food101_category}:")
        print(f"  Copying {num_to_copy} images...")
        
        copied = 0
        for i, source_img in enumerate(tqdm(source_images[:num_to_copy], desc=f"  {category}")):
            try:
                # Copy image
                dest_img = Path(images_dir) / f"{category.lower()}_food101_{i:04d}.jpg"
                shutil.copy2(source_img, dest_img)
                
                # Create label (full image bounding box)
                label_path = Path(labels_dir) / f"{category.lower()}_food101_{i:04d}.txt"
                with open(label_path, 'w') as f:
                    # Full image annotation
                    f.write(f"{info['class_id']} 0.5 0.5 0.9 0.9\n")
                
                copied += 1
            except Exception as e:
                print(f"    Error copying {source_img.name}: {e}")
                continue
        
        print(f"  ✓ Added {copied} images for {category}")
        total_added += copied
    
    return total_added

def download_from_kaggle_datasets(needs_supplement, images_dir, labels_dir):
    """
    Alternative: Download from Kaggle datasets
    Requires: pip install kaggle
    And Kaggle API credentials setup
    """
    
    try:
        import kaggle
    except ImportError:
        print("❌ Kaggle package not installed")
        print("   Install with: pip install kaggle")
        print("   Setup API: https://github.com/Kaggle/kaggle-api#api-credentials")
        return 0
    
    # Popular food datasets on Kaggle
    kaggle_datasets = {
        'Onion': 'saurabhshahane/onion-quality-dataset',
        'Garlic': None,  # Need to find
        'Lettuce': None,
        'Egg': 'saurabhshahane/egg-quality-dataset',
        'Butter': None,
        'Rice': 'muratkokludataset/rice-image-dataset',
        'Corn': 'saurabhshahane/corn-or-maize-leaf-disease-dataset'
    }
    
    total_added = 0
    
    for category, info in needs_supplement.items():
        dataset_name = kaggle_datasets.get(category)
        
        if not dataset_name:
            print(f"⊘ {category}: No Kaggle dataset mapped, skipping")
            continue
        
        print(f"\n{category}:")
        print(f"  Downloading from Kaggle: {dataset_name}")
        
        try:
            # Download dataset
            download_path = f"kaggle_data/{category.lower()}"
            kaggle.api.dataset_download_files(
                dataset_name, 
                path=download_path,
                unzip=True
            )
            
            # Find images in downloaded dataset
            dataset_path = Path(download_path)
            source_images = list(dataset_path.rglob('*.jpg')) + list(dataset_path.rglob('*.png'))
            
            num_to_copy = min(info['needed'], len(source_images))
            
            print(f"  Found {len(source_images)} images, copying {num_to_copy}...")
            
            copied = 0
            for i, source_img in enumerate(tqdm(source_images[:num_to_copy], desc=f"  {category}")):
                try:
                    # Copy image (convert PNG to JPG if needed)
                    dest_img = Path(images_dir) / f"{category.lower()}_kaggle_{i:04d}.jpg"
                    
                    if source_img.suffix.lower() == '.png':
                        from PIL import Image
                        img = Image.open(source_img).convert('RGB')
                        img.save(dest_img, 'JPEG')
                    else:
                        shutil.copy2(source_img, dest_img)
                    
                    # Create label
                    label_path = Path(labels_dir) / f"{category.lower()}_kaggle_{i:04d}.txt"
                    with open(label_path, 'w') as f:
                        f.write(f"{info['class_id']} 0.5 0.5 0.9 0.9\n")
                    
                    copied += 1
                except Exception as e:
                    continue
            
            print(f"  ✓ Added {copied} images for {category}")
            total_added += copied
            
            # Cleanup
            shutil.rmtree(download_path)
            
        except Exception as e:
            print(f"  ❌ Error downloading {dataset_name}: {e}")
            continue
    
    return total_added

def supplement_dataset_from_other_sources(
    labels_dir, 
    images_dir, 
    min_threshold=500,
    use_food101=True,
    use_kaggle=False
):
    """
    Main function to supplement from other datasets
    
    Args:
        labels_dir: Path to labels directory
        images_dir: Path to images directory  
        min_threshold: Minimum annotations per category
        use_food101: Whether to use Food-101 dataset
        use_kaggle: Whether to use Kaggle datasets (requires setup)
    """
    
    # Check what needs supplementing
    needs_supplement, _ = get_categories_needing_data(labels_dir, min_threshold)
    
    if not needs_supplement:
        print("✓ All categories have sufficient data!")
        return
    
    total_added = 0
    
    # Supplement from Food-101
    if use_food101:
        print("\n" + "="*70)
        print("SUPPLEMENTING FROM FOOD-101 DATASET")
        print("="*70)
        added = supplement_from_food101(needs_supplement, images_dir, labels_dir)
        total_added += added
    
    # Supplement from Kaggle
    if use_kaggle:
        print("\n" + "="*70)
        print("SUPPLEMENTING FROM KAGGLE DATASETS")
        print("="*70)
        added = download_from_kaggle_datasets(needs_supplement, images_dir, labels_dir)
        total_added += added
    
    print(f"\n{'='*70}")
    print(f"TOTAL IMAGES ADDED: {total_added}")
    print(f"{'='*70}")
    print("\n⚠️  IMPORTANT: Auto-generated labels use full-image bounding boxes.")
    print("   Consider manual annotation for better accuracy.")
    print("   Run distribution_check.py to verify the updated distribution.")

# ============================================================================
# USAGE
# ============================================================================

supplement_dataset_from_other_sources(
    labels_dir=r'C:/Users/User\Desktop/Master An2/Neural Networks/Ingredient Detection/ingredients_dataset/train/labels',
    images_dir=r'C:/Users/User\Desktop/Master An2/Neural Networks/Ingredient Detection/ingredients_dataset/train/images',
    min_threshold=500,
    use_food101=True,   # Download from Food-101 (free, no API needed)
    use_kaggle=False     # Set to True if you have Kaggle API setup
)
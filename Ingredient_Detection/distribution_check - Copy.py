import yaml
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt

def check_category_distribution(yaml_path, labels_dir):
    # Load YAML file
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    categories = config.get('names', config.get('classes', []))
    print(f"Found {len(categories)} categories in YAML")
    
    # Check labels directory
    label_path = Path(labels_dir)
    print(f"\nChecking labels directory: {label_path.absolute()}")
    print(f"Labels directory exists: {label_path.exists()}")
    
    if not label_path.exists():
        print("❌ Labels directory not found!")
        return
    
    label_files = list(label_path.glob('*.txt'))
    print(f"Found {len(label_files)} label files")
    
    if not label_files:
        print("❌ No .txt files found in labels directory!")
        return
    
    # Count annotations per category
    category_counts = Counter()
    
    for label_file in label_files:
        with open(label_file, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        class_id = int(line.split()[0])
                        category_counts[class_id] += 1
                    except (ValueError, IndexError):
                        continue
    
    # Display results
    print("\n" + "="*50)
    print("=== Category Distribution ===")
    print("="*50)
    total = sum(category_counts.values())
    print(f"Total annotations: {total}\n")
    
    for idx, category in enumerate(categories):
        count = category_counts.get(idx, 0)
        percentage = (count / total * 100) if total > 0 else 0
        print(f"{category:20s}: {count:5d} ({percentage:5.2f}%)")
    
    # Balance metrics
    if category_counts:
        counts = list(category_counts.values())
        avg = sum(counts) / len(counts)
        max_count = max(counts)
        min_count = min([c for c in counts if c > 0], default=max_count)
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        print(f"\n{'='*50}")
        print("=== Balance Metrics ===")
        print(f"{'='*50}")
        print(f"Average per category: {avg:.2f}")
        print(f"Max/Min ratio: {imbalance_ratio:.2f}")
        print(f"  (1.0 = perfectly balanced, >3.0 = significant imbalance)")
        
        # Visualize
        plt.figure(figsize=(14, 6))
        plt.bar(categories, [category_counts.get(i, 0) for i in range(len(categories))])
        plt.xlabel('Category', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.title('Annotation Distribution Across Categories', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('category_distribution.png', dpi=150)
        print("\n✓ Visualization saved as 'category_distribution.png'")

# Usage - point directly to your dataset/labels folder
check_category_distribution('ingredients_data.yaml', 'ingredients_dataset/train/labels')
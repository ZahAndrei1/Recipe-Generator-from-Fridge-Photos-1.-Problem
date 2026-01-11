import sys
import os
import numpy as np
import ast
from pathlib import Path

# Add paths for both modules
sys.path.append(os.path.join(os.getcwd(), 'Ingredient_Detection'))
sys.path.append(os.path.join(os.getcwd(), 'Recipe_Generator'))

from ultralytics import YOLO
from Generator import load_recipe_database, recommend_recipes
from Recipes import INGREDIENT_CLASSES

# Common pantry staples to assume are always available
PANTRY_STAPLES = ['salt', 'pepper', 'oil', 'water', 'flour', 'sugar']

def create_ingredient_vector(detected_ingredients, ingredient_classes):

    vector = np.zeros(len(ingredient_classes), dtype=int)
    
    for ingredient in detected_ingredients:
        ingredient_lower = ingredient.lower()
        for i, category in enumerate(ingredient_classes):
            category_lower = category.lower()
            if category_lower in ingredient_lower or ingredient_lower in category_lower:
                vector[i] = 1
                break
    
    return vector

def display_recipes(recommendations):

    print(f"\n{'='*70}")
    print("RECIPE RECOMMENDATIONS")
    print(f"{'='*70}\n")
    
    if not recommendations:
        print("❌ No recipes found matching your ingredients.")
        print("Try adding more ingredients or increase the max_missing parameter.")
        return
    
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['name']}")
        print(f"   Similarity Score: {rec['similarity']:.2%}")
        print(f"   Missing Ingredients: {rec['missing_count']}")
        
        if rec['missing_count'] > 0:
            print(f"   Need to add: {', '.join(rec['missing_ingredients'][:5])}")
            if len(rec['missing_ingredients']) > 5:
                print(f"                and {len(rec['missing_ingredients']) - 5} more...")
        
        print(f"\n   Ingredients:")
        try:
            if isinstance(rec['ingredients'], str):
                ingredients = ast.literal_eval(rec['ingredients'])
            else:
                ingredients = rec['ingredients']
            for ingr in ingredients[:10]:
                print(f"     • {ingr}")
            if len(ingredients) > 10:
                print(f"     ... and {len(ingredients) - 10} more")
        except:
            print(f"     {rec['ingredients']}")
        
        print(f"\n   Instructions:")
        try:
            if isinstance(rec['instructions'], str):
                steps = ast.literal_eval(rec['instructions'])
            else:
                steps = rec['instructions']
            for j, step in enumerate(steps[:5], 1):
                print(f"     {j}. {step}")
            if len(steps) > 5:
                print(f"     ... and {len(steps) - 5} more steps")
        except:
            print(f"     {rec['instructions']}")
        
        print(f"\n{'-'*70}\n")

def main(image_path, model_path='Ingredient_Detection/runs/ingredient_detection/demo5/weights/best.pt',
         recipe_db_path='Recipe_Generator/recipes_database.pkl',
         confidence_threshold=0.5, top_k=5, max_missing=2):
    print(f"\n{'#'*70}")
    print("INGREDIENT-TO-RECIPE RECOMMENDATION SYSTEM")
    print(f"{'#'*70}\n")
    
    # Step 1: Detect ingredients using YOLO
    print(f"\n{'='*70}")
    print("STEP 1: DETECTING INGREDIENTS")
    print(f"{'='*70}")
    
    model = YOLO(model_path)
    print(f"Loading model from: {model_path}")
    print(f"Analyzing image: {image_path}")
    
    results = model(image_path)
    
    detected_ingredients = []
    detection_details = {}
    
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            
            if confidence >= confidence_threshold:
                ingredient = model.names[class_id]
                detected_ingredients.append(ingredient)
                detection_details[ingredient] = confidence
    
    print(f"\nDetected {len(detected_ingredients)} ingredients:")
    for ingredient, confidence in detection_details.items():
        print(f"  ✓ {ingredient:20s} (confidence: {confidence:.2f})")
    
    for result in results:
        result.show()
    
    if not detected_ingredients:
        print("\n❌ No ingredients detected in the image.")
        print("Please try another image or lower the confidence threshold.")
        return
    
    # Step 2: Get recipe recommendations
    print(f"\n{'='*70}")
    print("STEP 2: GENERATING RECIPE RECOMMENDATIONS")
    print(f"{'='*70}")
    
    recipe_db = load_recipe_database(recipe_db_path)
    
    print("Creating ingredient vector...")
    detected_vector = create_ingredient_vector(detected_ingredients, INGREDIENT_CLASSES)
    
    print(f"Assuming available pantry staples: {', '.join(PANTRY_STAPLES)}")
    
    # Get initial recommendations using imported function
    initial_recommendations = recommend_recipes(detected_vector, recipe_db, top_k=top_k*3)
    
    # Filter by missing ingredients
    filtered_recommendations = []
    for rec in initial_recommendations:
        try:
            if isinstance(rec['ingredients'], str):
                recipe_ingr = ast.literal_eval(rec['ingredients'])
            else:
                recipe_ingr = rec['ingredients']
        except:
            continue
        
        missing = []
        for ingr in recipe_ingr:
            ingr_lower = ingr.lower()
            if any(staple in ingr_lower for staple in PANTRY_STAPLES):
                continue
            found = False
            for detected in detected_ingredients:
                if detected.lower() in ingr_lower or ingr_lower in detected.lower():
                    found = True
                    break
            if not found:
                missing.append(ingr)
        
        rec['missing_ingredients'] = missing
        rec['missing_count'] = len(missing)
        
        if len(missing) <= max_missing:
            filtered_recommendations.append(rec)
        
        if len(filtered_recommendations) >= top_k:
            break
    
    # Step 3: Display recommendations
    display_recipes(filtered_recommendations)
    
    print(f"\n{'#'*70}")
    print("PIPELINE COMPLETE")
    print(f"{'#'*70}\n")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Ingredient Detection and Recipe Recommendation')
    parser.add_argument('image_path', type=str, help='Path to the image with ingredients')
    parser.add_argument('--model', type=str, 
                       default='Ingredient_Detection/runs/ingredient_detection/demo/weights/best.pt',
                       help='Path to trained YOLO model weights')
    parser.add_argument('--recipe_db', type=str,
                       default='Recipe_Generator/recipes_database.pkl',
                       help='Path to recipe database')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Minimum confidence threshold for detection')
    parser.add_argument('--top_k', type=int, default=5,
                       help='Number of recipes to recommend')
    parser.add_argument('--max_missing', type=int, default=2,
                       help='Maximum number of missing ingredients allowed')
    
    args = parser.parse_args()
    
    main(
        image_path=args.image_path,
        model_path=args.model,
        recipe_db_path=args.recipe_db,
        confidence_threshold=args.confidence,
        top_k=args.top_k,
        max_missing=args.max_missing
    )
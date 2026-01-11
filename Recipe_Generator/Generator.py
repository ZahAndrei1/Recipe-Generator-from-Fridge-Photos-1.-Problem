import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import sys
import os
import ast

# Add parent directory to path to import from Ingredient Detection folder
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Ingredient_Detection'))
from inference import detect_ingredients

# IMPORTANT: This must match the model's training classes (from ingredients_data.yaml)
INGREDIENT_CLASSES = [
    'Apple', 'Banana', 'Orange', 'Tomato', 'Carrot',
    'Potato', 'Bread', 'Cheese', 'Broccoli', 'Strawberry',
    'Lemon', 'Cucumber', 'Onion', 'Garlic', 'Mushroom',
    'Lettuce', 'Egg', 'Chicken', 'Fish', 'Shrimp',
    'Milk', 'Butter', 'Rice', 'Pasta', 'Corn'
]

def load_recipe_database(pkl_path=None):
    """
    Load the preprocessed recipe database for matching.
    """
    if pkl_path is None:
        # Use path relative to this file
        pkl_path = os.path.join(os.path.dirname(__file__), 'recipes_database.pkl')

    if not os.path.exists(pkl_path):
        raise FileNotFoundError(
            f"Recipe database not found at: {pkl_path}\n"
            "Please run 'python Recipe_Generator/Recipes.py' first to generate the database."
        )

    with open(pkl_path, 'rb') as f:
        recipe_db = pickle.load(f)
    print(f"Loaded {len(recipe_db['names'])} recipes from database")
    return recipe_db
#def recommend_recipes(detected_ingredients_vector, recipe_db, top_k=5):
    """
    Recommend recipes based on detected ingredient vector using cosine similarity.
    
    Args:
        detected_ingredients_vector (np.array): binary array (length 54)
        recipe_db (dict): loaded recipe database with keys ['names', 'ingredients', 'vectors', 'instructions']
        top_k (int): number of top recipes to recommend
    
    Returns:
        List of dictionaries with recipe info and similarity score 
    """
    recipe_vectors = recipe_db['vectors']
    similarities = cosine_similarity(detected_ingredients_vector.reshape(1, -1), recipe_vectors)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]

    recommendations = []
    for idx in top_indices:
        recommendations.append({
            'name': recipe_db['names'][idx],
            'ingredients': recipe_db['ingredients'][idx],
            'instructions': recipe_db['instructions'][idx],
            'similarity': similarities[idx]
        })
    return recommendations

def recommend_recipes(detected_ingredients_vector, recipe_db, top_k=5, max_missing=2):
    """
    Recommend recipes, filtering out those that require too many extra ingredients.
    """
    recipe_vectors = recipe_db['vectors']
    
    # 1. Calculate Cosine Similarity
    similarities = cosine_similarity(detected_ingredients_vector.reshape(1, -1), recipe_vectors)[0]

    # 2. Calculate Missing Ingredients
    matches = recipe_vectors @ detected_ingredients_vector
    required_counts = recipe_vectors.sum(axis=1)
    missing_counts = required_counts - matches
    
    # 3. Apply Filter
    similarities[missing_counts > max_missing] = -1

    # Sort results
    top_indices = similarities.argsort()[-top_k:][::-1]

    recommendations = []
    for idx in top_indices:
        if similarities[idx] == -1:
            continue
            
        recommendations.append({
            'name': recipe_db['names'][idx],
            'ingredients': recipe_db['ingredients'][idx],
            'instructions': recipe_db['instructions'][idx], 
            'similarity': similarities[idx],
            'missing_count': int(missing_counts[idx]) # <--- This key must match the print statement
        })
        
    return recommendations


# Make sure you paste the NEW, LONG list of INGREDIENT_CLASSES at the top of Generator.py too!
# OR import it if you have it in a shared module.

if __name__ == "__main__":
    # Load the preprocessed recipe database 
    recipe_db = load_recipe_database()

    # Get the image path (relative to project root)
    project_root = os.path.join(os.path.dirname(__file__), '..')
    image_path = os.path.join(project_root, 'Ingredient_Detection', 'Fridge_Photo.jpeg')

    print(f"Detecting ingredients from: {image_path}")
    detected_items = detect_ingredients(image_path)
    print(f"Detected ingredients: {detected_items}")

    # Convert detected ingredients list to binary vector
    detected_vector = np.zeros(len(INGREDIENT_CLASSES), dtype=int)

    # Map detected ingredients to vector
    mapped_ingredients = []
    for item in detected_items:
        if item in INGREDIENT_CLASSES:
            idx = INGREDIENT_CLASSES.index(item)
            detected_vector[idx] = 1
            mapped_ingredients.append(item)

    print(f"\nMapped ingredients to vector: {mapped_ingredients}")
    print(f"Total ingredients detected: {int(detected_vector.sum())}\n")

    # Get recommendations
    print("=" * 60)
    print("FINDING RECIPE RECOMMENDATIONS...")
    print("=" * 60)
    recommendations = recommend_recipes(detected_vector, recipe_db, top_k=10, max_missing=2)

    detected_names = [name for idx, name in enumerate(INGREDIENT_CLASSES) if detected_vector[idx]]

    shown_count = 0
    filtered_count = 0
    for i, rec in enumerate(recommendations, 1):
        # Parse ingredient list if it's a string
        ingredient_list = rec['ingredients']
        if isinstance(ingredient_list, str):
            try:
                ingredient_list = ast.literal_eval(ingredient_list)
            except:
                ingredient_list = []

        # Count which recipe ingredients are actually in your fridge+staples
        matched = sum(any(cls.lower() in ingr.lower() for cls in detected_names) for ingr in ingredient_list)

        # Compute ingredient-coverage ratio
        coverage = matched / len(ingredient_list) if len(ingredient_list) > 0 else 0

        # Show only recipes with at least 10% ingredient coverage
        # (Adjust this threshold based on your needs: 0.0 = show all, 1.0 = perfect match only)
        if coverage < 0.1:
            filtered_count += 1
            continue

        shown_count += 1
        print(f"\n{shown_count}. {rec['name']}")
        print(f"   Similarity Score: {rec['similarity']:.2f}")
        print(f"   Missing Ingredients: {rec['missing_count']}")
        print(f"   Coverage: {coverage:.1%} ({matched}/{len(ingredient_list)} ingredients)")
        print(f"   Ingredients: {rec['ingredients']}")
        print(f"   Instructions: {str(rec['instructions'])[:200]}...")
        print("-" * 60)

    if shown_count == 0:
        print("\nNo recipes found matching your ingredient coverage criteria.")
        print("Try lowering the coverage threshold or detecting more ingredients.")

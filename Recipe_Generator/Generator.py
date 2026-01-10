import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
#from ingredients import INGREDIENT_CLASSES

# Now both files always use the exact same list!
INGREDIENT_CLASSES = [
    # --- Your Original 54 ---
    'Apple', 'Artichoke', 'Asparagus', 'Bagel', 'Banana', 'Bell pepper',
    'Bread', 'Broccoli', 'Burrito', 'Cabbage', 'Cake', 'Carrot', 'Cheese',
    'Cookie', 'Crab', 'Croissant', 'Cucumber', 'Doughnut', 'Egg',
    'French fries', 'Grape', 'Grapefruit', 'Guacamole', 'Hamburger',
    'Hot dog', 'Ice cream', 'Lemon', 'Lobster', 'Mango', 'Muffin',
    'Orange', 'Oyster', 'Pancake', 'Pasta', 'Peach', 'Pear', 'Pineapple',
    'Pizza', 'Pomegranate', 'Potato', 'Pretzel', 'Salad', 'Sandwich',
    'Shellfish', 'Shrimp', 'Strawberry', 'Submarine sandwich', 'Sushi',
    'Taco', 'Tart', 'Tomato', 'Waffle', 'Watermelon', 'Zucchini',
    
    # --- ESSENTIAL NEW ADDITIONS ---
    # Staples
    'Rice', 'Flour', 'Sugar', 'Salt', 'Pepper', 'Oil', 'Butter', 'Water', 
    'Vinegar', 'Yeast', 'Baking Powder', 'Milk', 'Cream', 'Yogurt',
    
    # Proteins
    'Chicken', 'Beef', 'Pork', 'Bacon', 'Sausage', 'Fish', 'Salmon', 
    'Tuna', 'Tofu', 'Beans', 'Lentils',
    
    # Common Veg/Aromatics (Crucial for filtering soups/stews)
    'Onion', 'Garlic', 'Ginger', 'Mushroom', 'Spinach', 'Corn', 'Peas',
    'Celery', 'Lettuce', 'Avocado', 'Lime', 'Cilantro', 'Basil', 'Parsley',
    
    # Sauces/Condiments
    'Soy Sauce', 'Ketchup', 'Mustard', 'Mayonnaise', 'Honey', 'Jam',
    'Chocolate', 'Vanilla', 'Nuts', 'Peanut Butter'
]

def load_recipe_database(pkl_path='Recipe_Generator\\recipes_database.pkl'):
    """
    Load the preprocessed recipe database for matching.
    """
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

    # DYNAMICALLY set vector size
    vector_size = len(INGREDIENT_CLASSES) 
    detected_vector = np.zeros(vector_size)

    # Set your detected/test ingredients
    for item in ['Apple', 'Carrot', 'Pasta', 'Tomato', 'Beef']:
        if item in INGREDIENT_CLASSES:
            detected_vector[INGREDIENT_CLASSES.index(item)] = 1

    # Add pantry staples so they don't count as "missing"
    staples = ['Salt', 'Pepper', 'Water', 'Oil', 'Sugar']
    for s in staples:
        if s in INGREDIENT_CLASSES:
            detected_vector[INGREDIENT_CLASSES.index(s)] = 1

    print(f"Vector size: {len(detected_vector)}")

    # Get recommendations
    recommendations = recommend_recipes(detected_vector, recipe_db, top_k=10, max_missing=2)

    shown_count = 0
    for i, rec in enumerate(recommendations, 1):
        ingredient_list = rec['ingredients']
        # Count which recipe ingredients are actually in your fridge+staples
        detected_names = [name for idx, name in enumerate(INGREDIENT_CLASSES) if detected_vector[idx]]
        matched = sum(any(cls.lower() in ingr.lower() for cls in detected_names) for ingr in ingredient_list)
        
        # Compute ingredient-coverage ratio
        coverage = matched / len(ingredient_list) if len(ingredient_list) > 0 else 0

        # Show only those where you have at least 50% of real ingredients
        if coverage < 0.0:
            continue

        shown_count += 1
        print(f"{shown_count}. {rec['name']}")
        print(f"   Similarity: {rec['similarity']:.2f}")
        print(f"   Missing Ingredients: {rec['missing_count']}")
        print(f"   Coverage: {coverage:.2f}")
        print(f"   Ingredients List: {rec['ingredients']}")
        print(f"   Instructions: {str(rec['instructions'])[:200]}...")
        print("-" * 50)
    
    if shown_count == 0:
        print("No recipes found matching your ingredient coverage criteria.")

import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def load_recipe_database(pkl_path='recipes_database.pkl'):
    """
    Load the preprocessed recipe database for matching.
    """
    with open(pkl_path, 'rb') as f:
        recipe_db = pickle.load(f)
    print(f"Loaded {len(recipe_db['names'])} recipes from database")
    return recipe_db

def recommend_recipes(detected_ingredients_vector, recipe_db, top_k=5):
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

if __name__ == "__main__":
    # Load the preprocessed recipe database 
    recipe_db = load_recipe_database()

    # Example detected ingredient vector (dummy: Apple and Carrot detected)
    detected_vector = np.zeros(54)
    detected_vector[0] = 1  # Apple
    detected_vector[11] = 1 # Carrot

    # Get recommendations
    recommendations = recommend_recipes(detected_vector, recipe_db, top_k=5)

    # Print recommendations
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['name']} (Similarity: {rec['similarity']:.2f})")
        print("Ingredients:", rec['ingredients'])
        print("Instructions:", rec['instructions'])
        print()

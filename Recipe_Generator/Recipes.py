
#https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions?resource=download
import pandas as pd
import numpy as np
import ast
import pickle
import kagglehub
import os, shutil

# Your 54 ingredient categories (from detector)
INGREDIENT_CLASSES = [
    'Apple', 'Artichoke', 'Asparagus', 'Bagel', 'Banana', 'Bell pepper',
    'Bread', 'Broccoli', 'Burrito', 'Cabbage', 'Cake', 'Carrot', 'Cheese',
    'Cookie', 'Crab', 'Croissant', 'Cucumber', 'Doughnut', 'Egg',
    'French fries', 'Grape', 'Grapefruit', 'Guacamole', 'Hamburger',
    'Hot dog', 'Ice cream', 'Lemon', 'Lobster', 'Mango', 'Muffin',
    'Orange', 'Oyster', 'Pancake', 'Pasta', 'Peach', 'Pear', 'Pineapple',
    'Pizza', 'Pomegranate', 'Potato', 'Pretzel', 'Salad', 'Sandwich',
    'Shellfish', 'Shrimp', 'Strawberry', 'Submarine sandwich', 'Sushi',
    'Taco', 'Tart', 'Tomato', 'Waffle', 'Watermelon', 'Zucchini'
]

def create_ingredient_vector(recipe_ingredients, ingredient_classes):
    """
    Convert recipe ingredients list string to a binary vector.
    
    Args:
        recipe_ingredients: string (list format) or list of ingredients
        ingredient_classes: list of ingredient category strings
    
    Returns:
        Numpy int array (length = number of ingredient classes)
    """
    vector = np.zeros(len(ingredient_classes), dtype=int)
    
    # Parse string representation of list to actual list
    try:
        if isinstance(recipe_ingredients, str):
            ingredients = ast.literal_eval(recipe_ingredients)
        else:
            ingredients = recipe_ingredients
    except:
        return vector  # Parsing failed, return zero vector
    
    # Check for presence of each ingredient category by substring match
    for i, category in enumerate(ingredient_classes):
        cat_lower = category.lower()
        for ingr in ingredients:
            ingr_lower = ingr.lower()
            if cat_lower in ingr_lower:
                vector[i] = 1
                break
            # Handle simple plurals
            if cat_lower.endswith('y'):
                if cat_lower[:-1] + 'ies' in ingr_lower:
                    vector[i] = 1
                    break
            elif cat_lower + 's' in ingr_lower:
                vector[i] = 1
                break
                
    return vector

def preprocess_food_com_dataset(csv_path, output_path='recipes_processed.pkl'):
    """
    Preprocess the Food.com raw recipes CSV for recipe matching.
    
    Args:
        csv_path: path to 'RAW_recipes.csv'
        output_path: path to save the processed pickle file
    
    Returns:
        DataFrame filtered with processed recipes and ingredient vectors
    """
    print("Loading dataset from", csv_path)
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} recipes")
    
    print("Creating ingredient vectors...")
    df['ingredient_vector'] = df['ingredients'].apply(
        lambda x: create_ingredient_vector(x, INGREDIENT_CLASSES)
    )
    
    print("Filtering recipes with at least one detectable ingredient...")
    df['has_ingredients'] = df['ingredient_vector'].apply(lambda x: x.sum() > 0)
    df_filtered = df[df['has_ingredients']].copy()
    print(f"Filtered down to {len(df_filtered)} recipes")
    
    df_filtered['ingredient_count'] = df_filtered['ingredient_vector'].apply(lambda x: x.sum())
    
    print("Saving processed recipes to", output_path)
    df_filtered.to_pickle(output_path)
    
    print("\nDataset statistics:")
    print("Total filtered recipes:", len(df_filtered))
    print(f"Average ingredient count per recipe: {df_filtered['ingredient_count'].mean():.2f}")
    
    # Ingredient occurrence counts
    ingredient_counts = np.zeros(len(INGREDIENT_CLASSES))
    for vec in df_filtered['ingredient_vector']:
        ingredient_counts += vec
    
    top_indices = ingredient_counts.argsort()[-10:][::-1]
    print("Top 10 ingredients by occurrence:")
    for idx in top_indices:
        print(f"  {INGREDIENT_CLASSES[idx]}: {int(ingredient_counts[idx])} recipes")
    
    return df_filtered

def save_recipe_database(df_filtered, output_path='recipes_database.pkl'):
    """
    Save the processed recipe data to a pickle database suitable for matching.
    
    Args:
        df_filtered: DataFrame with recipe data and 'ingredient_vector' column
        output_path: File path for saving the pickle database
    """
    data_to_save = {
        'names': df_filtered['name'].tolist(),
        'ingredients': df_filtered['ingredients'].tolist(),
        'vectors': np.vstack(df_filtered['ingredient_vector'].values),
        'instructions': df_filtered['steps'].tolist(),
        'cooking_time': df_filtered['minutes'].tolist()
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(data_to_save, f)
    
    print(f"Saved recipe database to {output_path}")

# Example usage:
if __name__ == "__main__":
    if not os.path.exists('Recipe_Generator\\RAW_recipes.csv'):
        path = kagglehub.dataset_download("shuyangli94/food-com-recipes-and-user-interactions")
        dest_folder = os.path.join(os.getcwd(), "Recipe_Generator")
        shutil.move(os.path.join(path, "RAW_recipes.csv"),os.path.join(dest_folder, "RAW_recipes.csv"))
        
    processed_df = preprocess_food_com_dataset('Recipe_Generator\\RAW_recipes.csv', 'Recipe_Generator\\recipes_processed.pkl')
    save_recipe_database(processed_df, 'Recipe_Generator\\recipes_database.pkl')

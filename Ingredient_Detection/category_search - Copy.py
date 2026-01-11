import pandas as pd

def search_openimages_classes():
    """Search OpenImages dataset for ingredient-related classes"""
    
    # Download class descriptions
    print("Downloading OpenImages class descriptions...")
    url = "https://storage.googleapis.com/openimages/v6/oidv6-class-descriptions.csv"
    df = pd.read_csv(url, names=['code', 'name'])
    
    print(f"Total classes in OpenImages: {len(df)}\n")
    
    # Search for missing categories
    search_terms = ['egg', 'rice', 'butter', 'corn', 'lettuce', 'onion', 'garlic']
    
    print("=" * 70)
    print("SEARCHING FOR MISSING INGREDIENTS")
    print("=" * 70)
    
    for term in search_terms:
        print(f"\n🔍 Searching for '{term.upper()}':")
        print("-" * 70)
        matches = df[df['name'].str.contains(term, case=False, na=False)]
        
        if not matches.empty:
            for _, row in matches.iterrows():
                print(f"  ✓ {row['name']:30s} -> {row['code']}")
        else:
            print(f"  ✗ No matches found for '{term}'")
    
    # Also search for broader food categories
    print("\n" + "=" * 70)
    print("SEARCHING FOR GENERAL FOOD-RELATED CLASSES")
    print("=" * 70)
    
    food_keywords = ['food', 'vegetable', 'fruit', 'dairy', 'grain', 'meat', 'seafood']
    
    for keyword in food_keywords:
        matches = df[df['name'].str.contains(keyword, case=False, na=False)]
        if not matches.empty:
            print(f"\n📦 '{keyword.upper()}' related classes ({len(matches)} found):")
            print("-" * 70)
            for _, row in matches.head(10).iterrows():  # Show first 10
                print(f"  • {row['name']:30s} -> {row['code']}")
            if len(matches) > 10:
                print(f"  ... and {len(matches) - 10} more")
    
    # Search for all current ingredients to verify
    print("\n" + "=" * 70)
    print("VERIFYING ALL YOUR INGREDIENTS")
    print("=" * 70)
    
    current_ingredients = ['Apple', 'Banana', 'Orange', 'Tomato', 'Carrot', 
                          'Potato', 'Bread', 'Cheese', 'Broccoli', 'Strawberry',
                          'Lemon', 'Cucumber', 'Onion', 'Garlic', 'Mushroom',
                          'Lettuce', 'Egg', 'Chicken', 'Fish', 'Shrimp',
                          'Milk', 'Butter', 'Rice', 'Pasta', 'Corn']
    
    found = []
    not_found = []
    
    for ingredient in current_ingredients:
        match = df[df['name'] == ingredient]
        if not match.empty:
            found.append((ingredient, match.iloc[0]['code']))
            print(f"✓ {ingredient:20s} -> {match.iloc[0]['code']}")
        else:
            not_found.append(ingredient)
            print(f"✗ {ingredient:20s} -> NOT FOUND")
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Found: {len(found)}/{len(current_ingredients)} ingredients")
    print(f"Missing: {len(not_found)} ingredients")
    
    if not_found:
        print(f"\nMissing ingredients: {', '.join(not_found)}")
        print("\n💡 Tip: Try searching for alternative names above!")
    
    # Save results to CSV for reference
    results_df = pd.DataFrame(found, columns=['Ingredient', 'Code'])
    results_df.to_csv('openimages_found_ingredients.csv', index=False)
    print("\n✓ Found ingredients saved to 'openimages_found_ingredients.csv'")

# Run the diagnostic
search_openimages_classes()
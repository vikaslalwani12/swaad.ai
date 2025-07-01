# meal_plan_generator.py

import pandas as pd
import random
import itertools
from constants import INDIAN_CUISINES, CARB_ITEMS, ROTI, RICE
from ingredient_utils import normalize_ingredient, jaccard_similarity

# Load the dataset
def load_recipes(file_path):
    return pd.read_csv(file_path)

# Filter recipes based on user ingredients
def filter_recipes_by_ingredients(recipes, ingredients):
    filtered_recipes = recipes[recipes['Cleaned-Ingredients'].apply(lambda x: all(item in x for item in ingredients))]
    return filtered_recipes

def clean_name(name):
    import re
    name = re.sub(r'<.*?>', '', str(name))  # Remove HTML tags
    name = re.sub(r'[^\w\s\-\(\)]', '', name)  # Remove special chars except - and ()
    return name.strip()

# Generate a meal plan based on age and health goals
def generate_meal_plan(recipes, user_ingredients, health_goals, goal):
    """
    Generate a meal plan that matches the user's nutrition targets and available ingredients,
    ensuring cuisine and dish variety across meals if possible.
    """
    meal_types = ['Breakfast', 'Lunch', 'Snack', 'Dinner']
    meal_plan = []
    used_indices = set()
    used_cuisines = set()
    used_dish_keywords = set()
    fallback_triggered = False
    def get_primary_dish_keyword(name):
        # Use first significant word (not stopwords) as dish keyword, e.g. 'Paneer', 'Biryani', etc.
        import re
        stopwords = {'the', 'with', 'and', 'of', 'in', 'on', 'a', 'an', 'recipe', 'curry', 'sabzi', 'dal', 'masala', 'stuffed', 'dry', 'gravy', 'tadka', 'tadkewali', 'ka', 'ke', 'ki', 'by', 'for', 'to', 'from'}
        words = re.split(r'\W+', name.lower())
        for w in words:
            if w and w not in stopwords:
                return w.capitalize()
        return name.split()[0] if name else ''
    for meal in meal_types:
        candidates = recipes.loc[~recipes.index.isin(used_indices)].copy()
        candidates['ingredient_overlap'] = candidates['Cleaned-Ingredients'].apply(
            lambda x: len(set(i.strip() for i in str(x).split(',')) & set(user_ingredients))
        )
        candidates = candidates[candidates['ingredient_overlap'] > 0]
        # Filter for cuisine and dish variety if possible
        diverse_candidates = candidates.copy()
        if len(candidates) > 0:
            if used_cuisines:
                diverse_candidates = diverse_candidates[~diverse_candidates['Cuisine'].str.lower().isin({c.lower() for c in used_cuisines})]
            if used_dish_keywords:
                diverse_candidates = diverse_candidates[~diverse_candidates['TranslatedRecipeName'].apply(lambda n: get_primary_dish_keyword(n) in used_dish_keywords)]
        # If not enough diverse options, fallback to original candidates
        if diverse_candidates.empty or len(diverse_candidates) < 2:
            diverse_candidates = candidates
            if not fallback_triggered and not candidates.empty:
                print("Limited diverse options available, showing best nutritional matches instead.")
                fallback_triggered = True
        if diverse_candidates.empty:
            continue
        # Score by closeness to macro targets for this meal (1/4th of daily target)
        target_cals = goal['calories'] / 4
        target_protein = goal['protein'] / 4
        target_carbs = goal['carbs'] / 4
        target_fat = goal['fat'] / 4
        diverse_candidates['score'] = (
            (diverse_candidates['Calories'] - target_cals).abs() +
            (diverse_candidates['Protein_g'] - target_protein).abs() +
            (diverse_candidates['Carbs_g'] - target_carbs).abs() +
            (diverse_candidates['Fat_g'] - target_fat).abs()
        )
        best = diverse_candidates.sort_values('score').iloc[0]
        used_indices.add(best.name)
        used_cuisines.add(str(best['Cuisine']))
        used_dish_keywords.add(get_primary_dish_keyword(str(best['TranslatedRecipeName'])))
        meal_plan.append({
            'meal': meal,
            'recipe_name': clean_name(best['TranslatedRecipeName']),
            'calories': best['Calories'],
            'protein': best['Protein_g'],
            'carbs': best['Carbs_g'],
            'fat': best['Fat_g'],
            'ingredients': best['Ingredients'],
            'cleaned_ingredients': best.get('Cleaned-Ingredients', ''),
            'instructions': best['TranslatedInstructions']
        })
    return meal_plan
# recommendation_engine.py
import os
import json
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from ingredient_utils import normalize_ingredient
from constants import INDIAN_CUISINES

def cluster_recipes(recipes, n_clusters=6):
    recipes['ingredient_list'] = recipes['Cleaned-Ingredients'].apply(lambda s: [normalize_ingredient(i) for i in str(s).split(',') if i.strip()])
    recipes['ingredient_str'] = recipes['ingredient_list'].apply(lambda ings: ','.join(sorted(set(ings))))
    vectorizer = CountVectorizer(token_pattern=r'[^,]+')
    X = vectorizer.fit_transform(recipes['ingredient_str'])
    k = min(n_clusters, len(recipes)) if len(recipes) >= n_clusters else 1
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    recipes['cluster'] = kmeans.fit_predict(X)
    return recipes

def load_upvote_history(recipes, upvote_file):
    preferred_clusters = set()
    if os.path.exists(upvote_file):
        try:
            with open(upvote_file, 'r', encoding='utf-8') as f:
                prev_upvotes = json.load(f)
            if isinstance(prev_upvotes, dict):
                for meal_type, recipe_name in prev_upvotes.items():
                    match = recipes[recipes['TranslatedRecipeName'].str.lower() == recipe_name.lower()]
                    if not match.empty:
                        cluster_num = match.iloc[0]['cluster']
                        preferred_clusters.add(cluster_num)
        except Exception:
            pass
    return preferred_clusters

def save_upvote_history(upvote_map, upvote_file):
    try:
        with open(upvote_file, 'w', encoding='utf-8') as f:
            json.dump(upvote_map, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False

def get_clusters_for_recipes(recipes_df, recipe_names):
    """
    Given a DataFrame and a list of recipe names, return a set of cluster labels for those recipes.
    """
    clusters = set()
    for name in recipe_names:
        row = recipes_df[recipes_df['TranslatedRecipeName'].str.lower() == name.strip().lower()]
        if not row.empty and 'cluster' in row.columns:
            clusters.update(row['cluster'].unique())
    return clusters

def generate_meal_plan_with_upvotes(recipes, user_ingredients, health_goals, targets, upvoted_clusters=None, last_recommended=None, diversity_ratio=0.7):
    """
    Prioritize recipes from upvoted clusters, ensure diversity, and avoid repeats.
    Fallback to macro heuristics if any macro is 0g or missing.
    """
    import numpy as np
    import pandas as pd
    # Heuristic macro mapping for fallback
    macro_heuristics = {
        'dal': {'protein': 8, 'carbs': 20, 'fat': 2, 'calories': 140},
        'paneer': {'protein': 14, 'carbs': 6, 'fat': 20, 'calories': 250},
        'biryani': {'protein': 8, 'carbs': 45, 'fat': 10, 'calories': 350},
        'khichdi': {'protein': 7, 'carbs': 35, 'fat': 5, 'calories': 220},
        'sandwich': {'protein': 7, 'carbs': 30, 'fat': 8, 'calories': 220},
        'idli': {'protein': 3, 'carbs': 15, 'fat': 0.5, 'calories': 70},
        'upma': {'protein': 5, 'carbs': 30, 'fat': 5, 'calories': 180},
        'rice': {'protein': 4, 'carbs': 45, 'fat': 0.5, 'calories': 200},
        'roti': {'protein': 3, 'carbs': 20, 'fat': 0.5, 'calories': 100},
        'sabzi': {'protein': 3, 'carbs': 10, 'fat': 5, 'calories': 80},
        'curd': {'protein': 4, 'carbs': 6, 'fat': 4, 'calories': 60},
        'egg': {'protein': 6, 'carbs': 1, 'fat': 5, 'calories': 70},
        'chana': {'protein': 7, 'carbs': 20, 'fat': 2, 'calories': 120},
        'bhurji': {'protein': 12, 'carbs': 5, 'fat': 15, 'calories': 200},
        'kurma': {'protein': 5, 'carbs': 15, 'fat': 18, 'calories': 220},
        'pulao': {'protein': 5, 'carbs': 40, 'fat': 8, 'calories': 220},
        'salad': {'protein': 2, 'carbs': 8, 'fat': 2, 'calories': 50},
        'soup': {'protein': 3, 'carbs': 8, 'fat': 2, 'calories': 60},
    }
    def macro_fallback(recipe_name, macro):
        for key, vals in macro_heuristics.items():
            if key in recipe_name.lower():
                return vals[macro]
        return 0
    if upvoted_clusters is None or len(upvoted_clusters) == 0:
        from meal_planner import generate_meal_plan
        return generate_meal_plan(recipes, user_ingredients, health_goals, targets)
    # Remove last recommended recipes
    if last_recommended:
        recipes = recipes[~recipes['TranslatedRecipeName'].isin(last_recommended)]
    # Split recipes by cluster
    in_cluster = recipes[recipes['cluster'].isin(upvoted_clusters)]
    out_cluster = recipes[~recipes['cluster'].isin(upvoted_clusters)]
    n_total = 4  # breakfast, lunch, snack, dinner
    n_in = int(np.ceil(n_total * diversity_ratio))
    n_out = n_total - n_in
    prioritized = in_cluster.sample(n=min(n_in, len(in_cluster)), random_state=None) if not in_cluster.empty else pd.DataFrame()
    others = out_cluster.sample(n=min(n_out, len(out_cluster)), random_state=None) if not out_cluster.empty else pd.DataFrame()
    selected = pd.concat([prioritized, others]).sample(frac=1, random_state=None)
    if len(selected) < n_total:
        needed = n_total - len(selected)
        extra = recipes[~recipes['TranslatedRecipeName'].isin(selected['TranslatedRecipeName'])].sample(n=needed, random_state=None)
        selected = pd.concat([selected, extra])
    meal_names = ['Breakfast', 'Lunch', 'Snack', 'Dinner']
    meal_plan = []
    for i, (_, row) in enumerate(selected.iterrows()):
        meal = meal_names[i] if i < len(meal_names) else f'Meal {i+1}'
        def safe_get(col, default=0):
            return row[col] if col in row and pd.notnull(row[col]) else default
        protein = safe_get('Protein_g')
        carbs = safe_get('Carbs_g')
        fat = safe_get('Fat_g')
        calories = safe_get('Calories')
        # Fallback if any macro is 0 or missing
        if not protein or protein == 0:
            protein = macro_fallback(row['TranslatedRecipeName'], 'protein')
        if not carbs or carbs == 0:
            carbs = macro_fallback(row['TranslatedRecipeName'], 'carbs')
        if not fat or fat == 0:
            fat = macro_fallback(row['TranslatedRecipeName'], 'fat')
        if not calories or calories == 0:
            calories = macro_fallback(row['TranslatedRecipeName'], 'calories')
        ingredients = safe_get('Ingredients', '')
        cleaned_ingredients = safe_get('Cleaned-Ingredients', '')
        instructions = row['Instructions'] if 'Instructions' in row else row.get('TranslatedInstructions', '')
        meal_plan.append({
            'meal': meal,
            'recipe_name': row['TranslatedRecipeName'],
            'calories': calories,
            'protein': protein,
            'carbs': carbs,
            'fat': fat,
            'ingredients': ingredients,
            'cleaned_ingredients': cleaned_ingredients,
            'url': row.get('RecipeUrl', '') or row.get('Recipe URL', ''),
            'instructions': instructions
        })
    return meal_plan

# ingredient_utils.py
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from constants import INDIAN_CUISINES

def normalize_ingredient(ing):
    ing = ing.lower()
    ing = re.sub(r"\s*\([^)]*\)", "", ing)
    ing = re.sub(r"\b(as needed|to taste|chopped|grated|sliced|diced|cubed|optional|as required|as per taste|as per your taste|as per taste and requirement|as per requirement|as per need|if available|if you like|if needed|if required|if using|if desired|if preferred|if possible|if you want|if you wish|if you have|if you prefer|or .*|you can use .*)\b", "", ing)
    ing = re.sub(r"\s+", " ", ing)
    ing = ing.strip().strip('.')
    return ing

def jaccard_similarity(set1, set2):
    if not set1 and not set2:
        return 1.0
    intersection = set1 & set2
    union = set1 | set2
    if not union:
        return 0.0
    return len(intersection) / len(union)

def get_best_recipe_matches(recipes, user_ingredients, veg_only=False, top_n=20, min_similarity=0.2):
    if 'Cleaned-Ingredients' not in recipes.columns:
        recipes['Cleaned-Ingredients'] = ''
    if 'Cuisine' not in recipes.columns:
        recipes['Cuisine'] = ''
    if veg_only:
        from constants import NON_VEG_KEYWORDS  # Changed from relative to absolute import
        def is_veg(row):
            name = str(row.get('TranslatedRecipeName', '')).lower()
            ings = str(row.get('Cleaned-Ingredients', '')).lower()
            return not any(k in name or k in ings for k in NON_VEG_KEYWORDS)
        recipes = recipes[recipes.apply(is_veg, axis=1)].copy()
    recipes['Cuisine'] = recipes['Cuisine'].fillna('').astype(str)
    is_indian = recipes['Cuisine'].str.lower().apply(lambda x: any(c in x for c in INDIAN_CUISINES))
    # Normalize user ingredients
    user_ings_set = set(normalize_ingredient(i) for i in user_ingredients if i.strip())
    # Compute Jaccard similarity for each recipe
    jaccard_sims = []
    for idx, row in recipes.iterrows():
        rec_ings = set(normalize_ingredient(i) for i in str(row.get('Cleaned-Ingredients', '')).split(',') if i.strip())
        sim = jaccard_similarity(user_ings_set, rec_ings)
        jaccard_sims.append(sim)
    recipes = recipes.copy()
    recipes['ingredient_similarity'] = jaccard_sims
    filtered = recipes[recipes['ingredient_similarity'] >= min_similarity]
    if filtered.shape[0] < top_n:
        filtered = recipes.sort_values('ingredient_similarity', ascending=False).head(top_n)
    else:
        filtered = filtered.sort_values('ingredient_similarity', ascending=False).head(top_n)
    filtered['is_indian'] = filtered['Cuisine'].str.lower().apply(lambda x: any(c in x for c in INDIAN_CUISINES))
    filtered = pd.concat([
        filtered[filtered['is_indian']],
        filtered[~filtered['is_indian']]
    ])
    return filtered.head(top_n)

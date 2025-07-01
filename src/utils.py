# meal_plan_generator.py

import pandas as pd
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from fitness_goal_classifier import train_fitness_goal_classifier, predict_fitness_goal

NON_VEG_KEYWORDS = ["chicken", "fish", "egg", "mutton", "lamb", "prawn", "shrimp", "meat", "beef", "pork", "trotter"]

INDIAN_CUISINES = [
    "indian", "south indian", "north indian", "punjabi", "gujarati", "maharashtrian", "bengali", "rajasthani", "tamil nadu", "andhra", "udupi", "chettinad", "kerala", "hyderabadi", "goan", "mughlai", "sindhi"
]
NON_INDIAN_CUISINES = [
    "mexican", "italian", "continental", "korean", "japanese", "greek", "thai", "chinese", "french", "american", "mediterranean", "spanish", "vietnamese", "lebanese", "turkish", "german", "russian"
]

COMMON_ESSENTIALS = set([
    'salt', 'water', 'oil', 'turmeric', 'cumin', 'garlic', 'ginger', 'onion', 'tomato', 'sugar', 'black pepper',
    'mustard seeds', 'asafoetida', 'hing', 'jeera', 'lemon', 'milk', 'curd', 'bread', 'wheat flour', 'atta', 'rice',
    'paneer', 'ghee', 'kasuri methi', 'curry leaves', 'bay leaf', 'cloves', 'cardamom', 'cinnamon', 'fenugreek',
    'star anise', 'nutmeg', 'mace', 'chili', 'vinegar', 'baking powder', 'baking soda', 'cornflour', 'corn starch',
    'butter', 'cream', 'yogurt', 'soya chunks', 'beetroot', 'coriander', 'coriander leaves', 'coriander powder',
    'green peas', 'peas', 'mustard oil', 'refined oil', 'sunflower oil', 'red chili', 'red chillies',
    'green chillies', 'green chilies', 'red chilli','red chilli powder', 'green chilli', 'chakundar', 'chaat masala'
])

# --- Ingredient Normalization Helper ---
def normalize_ingredient(ing):
    ing = ing.lower()
    ing = re.sub(r"\s*\([^)]*\)", "", ing)  # remove brackets
    # Remove common notes/phrases
    ing = re.sub(r"\b(as needed|to taste|chopped|grated|sliced|diced|cubed|optional|as required|as per taste|as per your taste|as per taste and requirement|as per requirement|as per need|if available|if you like|if needed|if required|if using|if desired|if preferred|if possible|if you want|if you wish|if you have|if you prefer|or .*|you can use .*)\b", "", ing)
    ing = re.sub(r"\s+", " ", ing)  # remove extra spaces
    ing = ing.strip().strip('.')
    return ing

# --- Ingredient Matching with TF-IDF Cosine Similarity ---
def get_best_recipe_matches(recipes, user_ingredients, veg_only=False, top_n=20, min_similarity=0.2):
    # Fallback for missing 'Cleaned-Ingredients' and 'Cuisine'
    if 'Cleaned-Ingredients' not in recipes.columns:
        recipes['Cleaned-Ingredients'] = ''
    if 'Cuisine' not in recipes.columns:
        recipes['Cuisine'] = ''
    # Vegetarian filter: exclude recipes with non-veg keywords if veg_only is True
    if veg_only:
        non_veg_keywords = [
            "chicken", "mutton", "egg", "fish", "prawn", "meat", "lamb", "beef", "pork", "shrimp", "trotter"
        ]
        def is_veg(row):
            name = str(row.get('TranslatedRecipeName', '')).lower()
            ings = str(row.get('Cleaned-Ingredients', '')).lower()
            return not any(k in name or k in ings for k in non_veg_keywords)
        recipes = recipes[recipes.apply(is_veg, axis=1)].copy()
    # Prefer Indian cuisines
    recipes['Cuisine'] = recipes['Cuisine'].fillna('').astype(str)
    is_indian = recipes['Cuisine'].str.lower().apply(lambda x: any(c in x for c in INDIAN_CUISINES))
    indian_recipes = recipes[is_indian].copy()
    non_indian_recipes = recipes[~is_indian].copy()
    # Build corpus for TF-IDF with ngram_range=(1,2)
    corpus = recipes['Cleaned-Ingredients'].fillna('').astype(str).tolist()
    user_str = ','.join(user_ingredients)
    tfidf = TfidfVectorizer(token_pattern=r'[^,]+', ngram_range=(1,2))
    tfidf_matrix = tfidf.fit_transform(corpus + [user_str])
    user_vec = tfidf_matrix[-1]
    recipe_vecs = tfidf_matrix[:-1]
    sims = cosine_similarity(user_vec, recipe_vecs).flatten()
    recipes = recipes.copy()
    recipes['ingredient_similarity'] = sims
    # Filter by minimum similarity threshold, but always keep at least top_n if possible
    filtered = recipes[recipes['ingredient_similarity'] >= min_similarity]
    if filtered.shape[0] < top_n:
        filtered = recipes.sort_values('ingredient_similarity', ascending=False).head(top_n)
    else:
        filtered = filtered.sort_values('ingredient_similarity', ascending=False).head(top_n)
    # Prioritize Indian recipes in the filtered set
    filtered['is_indian'] = filtered['Cuisine'].str.lower().apply(lambda x: any(c in x for c in INDIAN_CUISINES))
    filtered = pd.concat([
        filtered[filtered['is_indian']],
        filtered[~filtered['is_indian']]
    ])
    return filtered.head(top_n)

# Nutrition for staples
ROTI = {"calories": 100, "protein": 3, "carbs": 20, "fat": 0.5}
RICE = {"calories": 200, "protein": 4, "carbs": 45, "fat": 0.5}
CARB_ITEMS = ["roti", "rice", "paratha", "poha", "upma", "idli", "dosa", "thepla", "chapati", "bread"]
PROTEIN_ITEMS = ["dal", "paneer", "chana", "rajma", "sprouts", "curd", "yogurt", "tofu", "soy", "moong", "chickpea", "lentil", "sambar", "kadhi"]
SABZI_KEYWORDS = [
    "sabzi", "curry", "masala", "bhaji", "kootu", "poriyal", "thoran", "usili", "gobi", "bhindi", "aloo", "baingan", "cauliflower", "peas", "matar"
]
SNACK_KEYWORDS = ["snack", "cutlet", "pakora", "samosa", "chop", "vada", "tikki", "bhajiya", "fritter"]
CONDIMENT_KEYWORDS = ["raita", "chutney", "pickle", "papad", "salad", "buttermilk", "lassi", "tea", "coffee", "juice"]
SNACK_EXCLUDE_KEYWORDS = [
    "masala", "powder", "chutney", "pickle", "paste", "seasoning", "condiment", "spice mix", "gravy", "sauce"
]

def load_recipes(filepath):
    return pd.read_csv(filepath)

def is_main_course(name, ingredients):
    name = name.lower()
    ing = ','.join(ingredients).lower()
    # Exclude snacks/condiments
    if any(k in name for k in SNACK_KEYWORDS + CONDIMENT_KEYWORDS):
        return False
    # Must have a carb and a protein/veg
    has_carb = any(c in name or c in ing for c in CARB_ITEMS)
    has_protein = any(p in name or p in ing for p in PROTEIN_ITEMS + SABZI_KEYWORDS)
    return has_carb and has_protein

def is_sabzi_or_dal(name):
    name = name.lower()
    return any(k in name for k in SABZI_KEYWORDS + ["dal", "kadhi", "sambar", "chana", "rajma", "moong", "lentil"])

# --- Ingredient Matching with Jaccard Similarity ---
def jaccard_similarity(set1, set2):
    """
    Compute Jaccard similarity between two sets.
    Returns a float between 0 and 1.
    """
    if not set1 and not set2:
        return 1.0
    intersection = set1 & set2
    union = set1 | set2
    if not union:
        return 0.0
    return len(intersection) / len(union)

def get_best_recipe_matches(recipes, user_ingredients, veg_only=False, top_n=20, min_similarity=0.2):
    # Fallback for missing 'Cleaned-Ingredients' and 'Cuisine'
    if 'Cleaned-Ingredients' not in recipes.columns:
        recipes['Cleaned-Ingredients'] = ''
    if 'Cuisine' not in recipes.columns:
        recipes['Cuisine'] = ''
    # Vegetarian filter: exclude recipes with non-veg keywords if veg_only is True
    if veg_only:
        non_veg_keywords = [
            "chicken", "mutton", "egg", "fish", "prawn", "meat", "lamb", "beef", "pork", "shrimp", "trotter"
        ]
        def is_veg(row):
            name = str(row.get('TranslatedRecipeName', '')).lower()
            ings = str(row.get('Cleaned-Ingredients', '')).lower()
            return not any(k in name or k in ings for k in non_veg_keywords)
        recipes = recipes[recipes.apply(is_veg, axis=1)].copy()
    # Prefer Indian cuisines
    recipes['Cuisine'] = recipes['Cuisine'].fillna('').astype(str)
    is_indian = recipes['Cuisine'].str.lower().apply(lambda x: any(c in x for c in INDIAN_CUISINES))
    indian_recipes = recipes[is_indian].copy()
    non_indian_recipes = recipes[~is_indian].copy()
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
    # Filter by minimum similarity threshold, but always keep at least top_n if possible
    filtered = recipes[recipes['ingredient_similarity'] >= min_similarity]
    if filtered.shape[0] < top_n:
        filtered = recipes.sort_values('ingredient_similarity', ascending=False).head(top_n)
    else:
        filtered = filtered.sort_values('ingredient_similarity', ascending=False).head(top_n)
    # Prioritize Indian recipes in the filtered set
    filtered['is_indian'] = filtered['Cuisine'].str.lower().apply(lambda x: any(c in x for c in INDIAN_CUISINES))
    filtered = pd.concat([
        filtered[filtered['is_indian']],
        filtered[~filtered['is_indian']]
    ])
    return filtered.head(top_n)

# --- Explain recipe recommendation ---
def explain_recommendation(recipe, user_ingredients):
    recipe_ings = set(normalize_ingredient(i) for i in str(recipe.get('Cleaned-Ingredients', '')).split(',') if i.strip())
    user_ings = set(normalize_ingredient(i) for i in user_ingredients if i.strip())
    overlap = recipe_ings & user_ings
    percent = int(100 * len(overlap) / max(1, len(recipe_ings)))
    jaccard = jaccard_similarity(user_ings, recipe_ings)
    if len(overlap) > 0:
        overlap_str = ', '.join(sorted(overlap))
        reason = "High match with your ingredients"
    else:
        overlap_str = ""
        reason = "Chosen for nutritional balance"
    if percent < 10:
        reason = "Based on nutritional needs, as ingredient match was low."
    if overlap_str:
        macro = f"matched: {overlap_str}"
    elif recipe.get('EstimatedProtein', 0) > 10:
        macro = "high in protein"
    elif recipe.get('EstimatedCarbs', 0) > 30:
        macro = "carb-rich"
    elif recipe.get('EstimatedFat', 0) > 10:
        macro = "contains healthy fats"
    else:
        macro = "based on nutritional goals"
    return f"{reason}. {macro}. Jaccard Similarity: {jaccard:.2f}"

def calculate_nutrition_targets(weight, age, gender, activity_level, health_goal, height=170):
    # 1. Calculate BMR
    if gender.lower() == 'm' or gender.lower() == 'male':
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    else:
        bmr = 10 * weight + 6.25 * height - 5 * age - 161
    # 2. Activity factor
    activity_factors = {'sedentary': 1.2, 'lightly active': 1.375, 'moderately active': 1.55, 'very active': 1.725}
    activity_mult = activity_factors.get(activity_level.lower(), 1.2)
    tdee = bmr * activity_mult
    # 3. Protein
    if health_goal == 'muscle_gain':
        protein = 2.0 * weight
    elif health_goal == 'weight_loss':
        protein = 1.6 * weight
    else:
        protein = 1.8 * weight
    protein_kcal = protein * 4
    # 4. Fat (27% of TDEE)
    fat = 0.27 * tdee / 9
    fat_kcal = fat * 9
    # 5. Carbs (remaining)
    carbs_kcal = tdee - (protein_kcal + fat_kcal)
    carbs = carbs_kcal / 4
    return {
        'calories': int(tdee),
        'protein': int(protein),
        'carbs': int(carbs),
        'fat': int(fat),
        'protein_kcal': int(protein_kcal),
        'fat_kcal': int(fat_kcal),
        'carbs_kcal': int(carbs_kcal)
    }

def generate_meal_plan(recipes, user_ingredients, health_goals, goal):
    import itertools
    # Fallback for missing 'Cuisine'
    if 'Cuisine' not in recipes.columns:
        recipes['Cuisine'] = ''
    meal_types = ['Breakfast', 'Lunch', 'Dinner']
    meal_limits = {
        "weight_loss": (300, 450),
        "muscle_gain": (500, 750),
        "maintenance": (400, 600)
    }
    min_cal, max_cal = meal_limits.get(health_goals, (400, 600))
    # Filter for each meal type, prioritize Indian cuisines and avoid repeats
    meal_candidates = {}
    for meal_type in meal_types:
        candidates = recipes[(recipes['EstimatedCalories'] >= min_cal-100) & (recipes['EstimatedCalories'] <= max_cal+100)]
        candidates = candidates[(candidates['EstimatedProtein'] > 0) & (candidates['EstimatedCarbs'] > 0) & (candidates['EstimatedCalories'] > 0)]
        candidates = candidates.copy()
        candidates['prot_cal_ratio'] = candidates['EstimatedProtein'] / (candidates['EstimatedCalories'] + 1)
        # Prefer Indian cuisines
        candidates['is_indian'] = candidates['Cuisine'].str.lower().apply(lambda x: any(c in x for c in INDIAN_CUISINES))
        candidates = pd.concat([
            candidates[candidates['is_indian']],
            candidates[~candidates['is_indian']]
        ])
        meal_candidates[meal_type] = candidates.head(10)  # more variety
    # Try all combinations, avoid same meal or cuisine for all
    best_score = float('inf')
    best_combo = None
    best_total = None
    combos = list(itertools.product(
        meal_candidates['Breakfast'].to_dict('records'),
        meal_candidates['Lunch'].to_dict('records'),
        meal_candidates['Dinner'].to_dict('records')
    ))
    for b, l, d in combos:
        names = {b['TranslatedRecipeName'], l['TranslatedRecipeName'], d['TranslatedRecipeName']}
        cuisines = {b.get('Cuisine', '').lower(), l.get('Cuisine', '').lower(), d.get('Cuisine', '').lower()}
        if len(names) < 3:
            continue  # skip duplicate meals
        # Optionally, avoid all three meals from same cuisine
        if len(cuisines) < 2:
            continue
        # Score: balance ingredient similarity and nutrition
        meals = []
        for meal_type, rec in zip(meal_types, [b, l, d]):
            meal = {
                'meal': meal_type,
                'recipe_name': rec['TranslatedRecipeName'],
                'calories': float(rec['EstimatedCalories']),
                'protein': float(rec['EstimatedProtein']),
                'carbs': float(rec['EstimatedCarbs']),
                'fat': float(rec['EstimatedFat']),
                'instructions': deduplicate_instructions(rec['TranslatedInstructions']),
                'url': rec['URL'],
                'cuisine': rec.get('Cuisine', ''),
                'Cleaned-Ingredients': rec.get('Cleaned-Ingredients', ''),
                'Ingredients': rec.get('Ingredients', '')
            }
            meal, staple_added = add_staple(meal, meal_type, user_ingredients, {})
            if not is_valid_meal(meal):
                break
            meals.append(meal)
        if len(meals) < 3:
            continue
        # Score: combine macro difference and ingredient similarity
        macro_score, total = score_meal_combo(meals, goal)
        ing_score = -sum([rec.get('ingredient_similarity', 0) for rec in [b, l, d]])  # higher similarity = lower score
        score = macro_score + ing_score * 100  # weight ingredient match
        if score < best_score:
            best_score = score
            best_combo = meals
            best_total = total
    if best_combo:
        return best_combo
    else:
        print("⚠️ Could not find a good meal plan. Try adding more ingredients.")
        return []

def main():
    filepath = 'c:\\Users\\lalwa\\Desktop\\AIML Lab SEE\\indian_recipes_enriched_with_per_person_nutrition.csv'
    recipes = load_recipes(filepath)

    user_ingredients = [i.strip().lower() for i in input("Enter your available ingredients (comma-separated): ").split(',')]
    weight = float(input("Enter your weight (kg): "))
    age = int(input("Enter your age: "))
    # Gender normalization
    gender_input = input("Enter your gender (M/F): ").strip().lower()
    gender_map = {'m': 'Male', 'male': 'Male', 'f': 'Female', 'female': 'Female'}
    gender = gender_map.get(gender_input)
    if gender is None:
        print("Invalid gender. Please enter 'M' or 'F'.")
        return
    # Activity level normalization
    activity_level_input = input("Enter your activity level (sedentary/lightly active/moderately active/very active): ").strip().lower()
    activity_map = {
        'sedentary': 'Sedentary',
        'light': 'Lightly Active',
        'lightly active': 'Lightly Active',
        'moderate': 'Moderately Active',
        'moderately active': 'Moderately Active',
        'active': 'Very Active',
        'very active': 'Very Active'
    }
    activity_level = activity_map.get(activity_level_input)
    if activity_level is None:
        print("Invalid activity level. Please enter one of: sedentary, lightly active, moderately active, very active.")
        return
    # Dietary preference normalization
    dietary_pref_input = input("Enter your dietary preference (Omnivore/Vegetarian/Vegan): ").strip().lower()
    dietary_map = {'omnivore': 'Omnivore', 'vegetarian': 'Vegetarian', 'vegan': 'Vegan'}
    dietary_pref = dietary_map.get(dietary_pref_input)
    if dietary_pref is None:
        print("Invalid dietary preference. Please enter Omnivore, Vegetarian, or Vegan.")
        return
    height_in = input("Enter your height in cm (or press Enter for 170): ").strip()
    height = float(height_in) if height_in else 170

    # New: Ask for preferred Indian cuisine
    preferred_cuisine = input("""
Choose a preferred Indian cuisine (optional):
Options: North Indian, South Indian, Gujarati, Punjabi, Bengali, Maharashtrian, Rajasthani, Kashmiri, Chettinad, Andhra, Tamil Nadu, Kerala, Bihari, Assamese, Hyderabadi, Lucknowi, Goan, Sindhi, Coorg, Malvani, Mangalorean, Oriya Recipes, North Karnataka, South Karnataka, Karnataka, Awadhi, Konkan, Kongunadu, Uttarakhand-North Kumaon, Jharkhand, Nagaland
Type one of the above or press Enter to skip:
""").strip().lower()

    # Predict fitness goal using Naive Bayes classifier
    # Robust path for nutrition_dataset.csv
    dataset_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'nutrition_dataset.csv')
    clf, encoders, le_goal = train_fitness_goal_classifier(dataset_path)
    predicted_goal = predict_fitness_goal(
        clf, encoders, le_goal, age, gender, height, weight, activity_level, dietary_pref
    )
    print(f"Predicted fitness goal: {predicted_goal}")
    # Map predicted goal to internal code
    health_goals = predicted_goal.lower().replace(' ', '_')

    goal = calculate_nutrition_targets(weight, age, gender, activity_level, health_goals, height)
    print(f"\nYour personalized daily targets:")
    print(f"Calories: {goal['calories']} kcal | Protein: {goal['protein']}g | Carbs: {goal['carbs']}g | Fat: {goal['fat']}g\n")
    print(f"Macro breakdown: Protein {goal['protein_kcal']} kcal ({goal['protein_kcal']*100//goal['calories']}%), Fat {goal['fat_kcal']} kcal ({goal['fat_kcal']*100//goal['calories']}%), Carbs {goal['carbs_kcal']} kcal ({goal['carbs_kcal']*100//goal['calories']}%)\n")

    # --- TF-IDF ingredient matching and veg filter ---
    veg_only = (dietary_pref == 'Vegetarian' or dietary_pref == 'Vegan')

    # Cuisine filtering before recipe ranking
    filtered_recipes = recipes.copy()
    if preferred_cuisine:
        cuisine_mask = filtered_recipes['Cuisine'].astype(str).str.strip().str.lower() == preferred_cuisine
        cuisine_recipes = filtered_recipes[cuisine_mask]
        if cuisine_recipes.empty:
            print("No recipes found for this cuisine. Showing results from all cuisines instead.")
        else:
            filtered_recipes = cuisine_recipes

    filtered_recipes = get_best_recipe_matches(filtered_recipes, user_ingredients, veg_only=veg_only, top_n=30, min_similarity=0.2)
    if filtered_recipes.empty:
        print("⚠️ Not enough ingredients for a complete meal plan.")
        return

    meal_plan = generate_meal_plan(filtered_recipes, user_ingredients, health_goals, goal)

    total = {"calories": 0, "protein": 0, "carbs": 0, "fat": 0}
    used_names = []
    staple_summary = {}
    staple_log = set()
    for meal in meal_plan:
        recipe_name = meal.get('recipe_name') or meal.get('TranslatedRecipeName') or meal.get('RecipeName') or 'Unknown Recipe'
        protein = float(meal.get('protein', meal.get('Protein', meal.get('EstimatedProtein', 0))))
        calories = float(meal.get('calories', meal.get('Calories', meal.get('EstimatedCalories', 0))))
        carbs = float(meal.get('carbs', meal.get('Carbs', meal.get('EstimatedCarbs', 0))))
        fat = float(meal.get('fat', meal.get('Fat', meal.get('EstimatedFat', 0))))
        meal['protein'] = protein
        meal['calories'] = calories
        meal['carbs'] = carbs
        meal['fat'] = fat
        used_names.append(recipe_name)
        tag = get_meal_tag(meal)
        summary = get_instruction_summary(meal.get('instructions', meal.get('Instructions', '')))
        reason = explain_recommendation(meal, user_ingredients)
        staple_note = staple_summary.get(meal.get('meal', 'Meal'), "")
        if staple_note and (meal.get('meal', 'Meal'), staple_note) not in staple_log:
            print(f"🍽️ Added {staple_note} to {meal.get('meal', 'Meal')}.")
            staple_log.add((meal.get('meal', 'Meal'), staple_note))
        print(f"{meal.get('meal', 'Meal')}: {recipe_name} {tag} {staple_note}")
        print(f"Serving: 1")
        print(f"Summary: {summary}")
        print(f"Why: {reason}")
        print(f"Calories: {calories}, Protein: {protein}g, Carbs: {carbs}g, Fat: {fat}g")
        print(f"Recipe URL: {meal.get('url', meal.get('URL', ''))}\n")
        total["calories"] += calories
        total["protein"] += protein
        total["carbs"] += carbs
        total["fat"] += fat

    snack_needed = (total["calories"] < goal["calories"] - 150) or (total["protein"] < goal["protein"] - 10)
    if snack_needed:
        snack = select_snack(recipes, user_ingredients, used_names, total, goal)
        if snack:
            print(f"Snack: {snack['recipe_name']} (Calories: {snack['calories']}, Protein: {snack['protein']}g, Carbs: {snack['carbs']}g, Fat: {snack['fat']}g)")
            print(f"Summary: {get_instruction_summary(snack['instructions'])}")
            print(f"Recipe URL: {snack['url']}\n")
            total["calories"] += snack["calories"]
            total["protein"] += snack["protein"]
            total["carbs"] += snack["carbs"]
            total["fat"] += snack["fat"]

    # If meal plan total is off by >500 kcal, auto-suggest fillers
    kcal_gap = goal['calories'] - total['calories']
    if abs(kcal_gap) > 500:
        if kcal_gap > 0:
            print(f"\n⚡ You are {kcal_gap} kcal under your target. Suggestions: add a glass of milk, banana, sprouts, or a smoothie.")
        else:
            print(f"\n⚡ You are {abs(kcal_gap)} kcal over your target. Consider reducing portion size or skipping a snack.")

    print("📈 Daily Summary:")
    print(f"Total: {total['calories']} kcal / {total['protein']}g protein / {total['carbs']}g carbs / {total['fat']}g fat")
    print(f"Goal: {goal['calories']} kcal / {goal['protein']}g protein / {goal['carbs']}g carbs / {goal['fat']}g fat")
    feedback = []
    if total["calories"] > goal["calories"]:
        feedback.append(f"Over by {int(total['calories'] - goal['calories'])} kcal")
    elif total["calories"] < goal["calories"]:
        feedback.append(f"Under by {int(goal['calories'] - total['calories'])} kcal")
    if total["protein"] < goal["protein"]:
        feedback.append(f"You need {int(goal['protein'] - total['protein'])}g more protein. Suggest: curd + peanuts OR paneer salad.")
    elif total["protein"] > goal["protein"]:
        feedback.append(f"Over protein by {int(total['protein'] - goal['protein'])}g")
    if not feedback:
        print("✅ Perfectly matches your goal!")
    else:
        print("Feedback: " + "; ".join(feedback))

    # --- Shopping Basket Generator ---
    # Ingredient normalization helper for shopping list
    # Remove the inner definition of normalize_ingredient here (it is already defined at top-level)

    # 1. Collect all unique normalized ingredients from selected recipes (breakfast, lunch, dinner, snack)
    selected_meals = meal_plan.copy()
    if snack_needed and snack:
        selected_meals.append(snack)
    all_recipe_ings = set()
    for meal in selected_meals:
        ings_str = meal.get('Cleaned-Ingredients') or meal.get('Ingredients') or ''
        for ing in ings_str.split(','):
            ing_clean = normalize_ingredient(ing)
            if ing_clean:
                all_recipe_ings.add(ing_clean)
    all_recipe_ings = set(i for i in all_recipe_ings if i)

    # 2. Normalize user's available ingredients
    user_ings_set = set(normalize_ingredient(i) for i in user_ingredients if i.strip())
    user_ings_set = set(i for i in user_ings_set if i)

    # 3. Fuzzy/substring matching: for each recipe ingredient, check if any user ingredient is a substring or vice versa
    #    and ignore COMMON_ESSENTIALS
    shopping_list = []
    for rec_ing in all_recipe_ings:
        if rec_ing in COMMON_ESSENTIALS:
            continue
        found = False
        for user_ing in user_ings_set:
            if rec_ing in user_ing or user_ing in rec_ing:
                found = True
                break
        if not found:
            shopping_list.append(rec_ing)
    shopping_list = sorted(set(shopping_list))

    # 4. Display shopping list
    print("\n🛒 Shopping List (Ingredients you need to buy):")
    if shopping_list:
        for item in shopping_list:
            print("-", item)
    else:
        print("You have all the ingredients needed for your meal plan!")

    # Optionally offer user to see more suggestions
    more = input("\nWant suggestions to boost your meal? (y/n): ").strip().lower()
    if more == 'y':
        print("Try: 1 glass milk (120 kcal, 6g protein), 1 banana (100 kcal), 50g paneer (130 kcal, 8g protein), 2 tbsp peanuts (100 kcal, 5g protein)")

def add_staple(meal, meal_type, user_ingredients, staple_summary):
    # Only add staple if not already present in ingredients, and avoid repeated logs
    staple_added = None
    if is_sabzi_or_dal(meal['recipe_name']) and not any(c in meal['recipe_name'].lower() for c in CARB_ITEMS):
        if ("wheat flour" in user_ingredients or "atta" in user_ingredients or "roti" in user_ingredients) and staple_summary.get(meal_type) != 'roti':
            roti_count = 1 if meal_type == "Breakfast" else 2
            meal['calories'] += ROTI["calories"] * roti_count
            meal['protein'] += ROTI["protein"] * roti_count
            meal['carbs'] += ROTI["carbs"] * roti_count
            meal['fat'] += ROTI["fat"] * roti_count
            staple_summary[meal_type] = f"+{roti_count} roti(s)"
            staple_added = f"{roti_count} roti(s)"
        elif "rice" in user_ingredients and staple_summary.get(meal_type) != 'rice':
            rice_count = 1
            meal['calories'] += RICE["calories"] * rice_count
            meal['protein'] += RICE["protein"] * rice_count
            meal['carbs'] += RICE["carbs"] * rice_count
            meal['fat'] += RICE["fat"] * rice_count
            staple_summary[meal_type] = f"+{rice_count} cup rice"
            staple_added = f"{rice_count} cup rice"
    return meal, staple_added

def deduplicate_instructions(instructions):
    import re
    sentences = re.split(r'(?<=[.?!])\s+', instructions.strip())
    seen = set()
    cleaned = []
    for s in sentences:
        norm = ' '.join(s.lower().split())
        if norm and norm not in seen:
            cleaned.append(s.strip())
            seen.add(norm)
    result = '.'.join([c if c.endswith(('.', '!', '?')) else c + '.' for c in cleaned])
    return result.strip()

def get_instruction_summary(instructions):
    import re
    steps = re.split(r'(?<=[.?!])\s+', instructions.strip())
    summary = []
    for s in steps:
        s_clean = s.strip()
        if s_clean and len(' '.join(summary + [s_clean])) < 80:
            summary.append(s_clean)
        if len(summary) >= 2:
            break
    return "  ".join([s.split('.')[0] for s in summary if s])

def get_meal_tag(meal):
    if meal['protein'] >= 18:
        return " High Protein"
    if meal['calories'] <= 350:
        return " Low Calorie"
    if meal['protein'] >= 15 and meal['calories'] >= 400:
        return " Muscle Building"
    return " Balanced"

def is_valid_meal(meal):
    # Exclude meals with 0 protein or 0 carbs or 0 calories
    return meal['protein'] > 0 and meal['carbs'] > 0 and meal['calories'] > 0

def is_valid_snack(row):
    name = row['TranslatedRecipeName'].lower()
    ingredients = str(row['Cleaned-Ingredients']).lower()
    for k in SNACK_EXCLUDE_KEYWORDS:
        if k in name or k in ingredients:
            return False
    return True

def score_meal_combo(meals, goal):
    # Score by sum of absolute differences from goal macros
    total = {'calories': 0, 'protein': 0, 'carbs': 0, 'fat': 0}
    for meal in meals:
        total['calories'] += meal['calories']
        total['protein'] += meal['protein']
        total['carbs'] += meal['carbs']
        total['fat'] += meal['fat']
    score = (
        abs(goal['calories'] - total['calories']) +
        abs(goal['protein'] - total['protein']) * 10 +  # weight protein more
        abs(goal['carbs'] - total['carbs']) +
        abs(goal['fat'] - total['fat'])
    )
    return score, total

def select_snack(recipes, user_ingredients, used_names, total, goal):
    # Fallback for missing 'Cuisine'
    if 'Cuisine' not in recipes.columns:
        recipes['Cuisine'] = ''
    snack_cuisine = ["gujarati", "maharashtrian", "south indian", "tamil nadu", "kerala", "andhra", "udupi"]
    snack_candidates = recipes[
        (recipes['EstimatedCalories'] < 250) &
        (~recipes['TranslatedRecipeName'].isin(used_names))
    ]
    # Safely escape user ingredients for regex
    safe_ingredients = [re.escape(i) for i in user_ingredients if i]
    if safe_ingredients:
        pattern = '|'.join(safe_ingredients)
        snack_candidates = snack_candidates[
            snack_candidates['Cleaned-Ingredients'].str.contains(pattern, case=False, na=False)
        ]
    snack_candidates = snack_candidates[
        ~snack_candidates['TranslatedRecipeName'].str.lower().str.contains('|'.join(CONDIMENT_KEYWORDS))
    ]
    snack_candidates = snack_candidates[snack_candidates.apply(is_valid_snack, axis=1)]
    snack_candidates = snack_candidates[(snack_candidates['EstimatedProtein'] > 0) & (snack_candidates['EstimatedCarbs'] > 0) & (snack_candidates['EstimatedCalories'] > 0)]
    # Prefer lighter Indian snacks
    snack_candidates['is_light_indian'] = snack_candidates['Cuisine'].str.lower().apply(lambda x: any(c in x for c in snack_cuisine))
    snack_candidates = pd.concat([
        snack_candidates[snack_candidates['is_light_indian']],
        snack_candidates[~snack_candidates['is_light_indian']]
    ])
    if not snack_candidates.empty:
        for idx, snack in snack_candidates.sample(frac=1).iterrows():
            if is_valid_snack(snack):
                return {
                    'meal': 'Snack',
                    'recipe_name': snack['TranslatedRecipeName'],
                    'calories': float(snack['EstimatedCalories']),
                    'protein': float(snack['EstimatedProtein']),
                    'carbs': float(snack['EstimatedCarbs']),
                    'fat': float(snack['EstimatedFat']),
                    'instructions': deduplicate_instructions(snack['TranslatedInstructions']),
                    'url': snack['URL'],
                    'cuisine': snack.get('Cuisine', '')
                }
    return None

def score_meal(meal, user_ingredients, nutrition_target):
    """
    Returns a float score for a meal based on ingredient match and nutrition closeness to target.
    - ingredient_score: Jaccard similarity between user_ingredients and meal['ingredients_cleaned']
    - nutrition_score: 1 - (normalized macro diff)
    Final score = 0.5 * ingredient_score + 0.5 * nutrition_score
    """
    # Ingredient score
    user_set = set(normalize_ingredient(i) for i in user_ingredients if i.strip())
    meal_ings = meal.get('Cleaned-Ingredients') or meal.get('Ingredients') or meal.get('ingredients_cleaned') or ''
    meal_set = set(normalize_ingredient(i) for i in meal_ings.split(',') if i.strip())
    ingredient_score = jaccard_similarity(user_set, meal_set)

    # Nutrition score: sum of absolute macro differences, normalized
    macros = ['calories', 'protein', 'carbs', 'fat']
    meal_macros = [float(meal.get(m, 0)) for m in macros]
    target_macros = [float(nutrition_target.get(m, 1)) for m in macros]
    abs_diffs = [abs(m - t) for m, t in zip(meal_macros, target_macros)]
    # Normalize: divide by target, cap at 1.0
    norm_diffs = [min(a / t if t else 1.0, 1.0) for a, t in zip(abs_diffs, target_macros)]
    nutrition_score = 1.0 - (sum(norm_diffs) / len(norm_diffs))  # higher is better
    nutrition_score = max(0.0, min(nutrition_score, 1.0))

    # Final score
    final_score = 0.5 * ingredient_score + 0.5 * nutrition_score
    return final_score

# --- Recipe Recommendation System with Clustering and Upvote ---
def run_recipe_recommender():
    """
    Ingredient-aware, upvote-personalized recipe recommender with cluster preference.
    1. Asks for available ingredients and user details.
    2. Predicts health goal and computes macro targets.
    3. Loads upvote history and clusters recipes.
    4. Prefers (but does not force) recipes from upvoted clusters when generating today's meal plan.
    5. Suggests best matching recipes for each meal.
    6. At the end, allows user to upvote any meal(s) and saves mapping in last_upvotes.json.
    """
    import pandas as pd
    import os
    import json
    from fitness_goal_classifier import train_fitness_goal_classifier, predict_fitness_goal
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.cluster import KMeans

    # Step 1: Load dataset
    filepath = 'c:\\Users\\lalwa\\Desktop\\AIML Lab SEE\\indian_recipes_enriched_with_per_person_nutrition.csv'
    recipes = load_recipes(filepath)
    recipes = recipes.dropna(subset=['Cleaned-Ingredients'])
    recipes['ingredient_list'] = recipes['Cleaned-Ingredients'].apply(lambda s: [normalize_ingredient(i) for i in str(s).split(',') if i.strip()])
    recipes['ingredient_str'] = recipes['ingredient_list'].apply(lambda ings: ','.join(sorted(set(ings))))

    # Step 2: Take user inputs
    user_ingredients = [normalize_ingredient(i) for i in input("Enter your available ingredients (comma-separated): ").split(',') if i.strip()]
    weight = float(input("Enter your weight (kg): "))
    age = int(input("Enter your age: "))
    gender_input = input("Enter your gender (M/F): ").strip().lower()
    gender_map = {'m': 'Male', 'male': 'Male', 'f': 'Female', 'female': 'Female'}
    gender = gender_map.get(gender_input)
    if gender is None:
        print("Invalid gender. Please enter 'M' or 'F'.")
        return
    activity_level_input = input("Enter your activity level (sedentary/lightly active/moderately active/very active): ").strip().lower()
    activity_map = {
        'sedentary': 'Sedentary',
        'light': 'Lightly Active',
        'lightly active': 'Lightly Active',
        'moderate': 'Moderately Active',
        'moderately active': 'Moderately Active',
        'active': 'Very Active',
        'very active': 'Very Active'
    }
    activity_level = activity_map.get(activity_level_input)
    if activity_level is None:
        print("Invalid activity level. Please enter one of: sedentary, lightly active, moderately active, very active.")
        return
    dietary_pref_input = input("Enter your dietary preference (Omnivore/Vegetarian/Vegan): ").strip().lower()
    dietary_map = {'omnivore': 'Omnivore', 'vegetarian': 'Vegetarian', 'vegan': 'Vegan'}
    dietary_pref = dietary_map.get(dietary_pref_input)
    if dietary_pref is None:
        print("Invalid dietary preference. Please enter Omnivore, Vegetarian, or Vegan.")
        return
    height_in = input("Enter your height in cm (or press Enter for 170): ").strip()
    height = float(height_in) if height_in else 170
    preferred_cuisine = input("""
Choose a preferred Indian cuisine (optional):
Options: North Indian, South Indian, Gujarati, Punjabi, Bengali, Maharashtrian, Rajasthani, Kashmiri, Chettinad, Andhra, Tamil Nadu, Kerala, Bihari, Assamese, Hyderabadi, Lucknowi, Goan, Sindhi, Coorg, Malvani, Mangalorean, Oriya Recipes, North Karnataka, South Karnataka, Karnataka, Awadhi, Konkan, Kongunadu, Uttarakhand-North Kumaon, Jharkhand, Nagaland
Type one of the above or press Enter to skip:
""").strip().lower()

    # Step 3: Predict fitness goal
    dataset_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'nutrition_dataset.csv')
    clf, encoders, le_goal = train_fitness_goal_classifier(dataset_path)
    predicted_goal = predict_fitness_goal(
        clf, encoders, le_goal, age, gender, height, weight, activity_level, dietary_pref
    )
    print(f"Predicted fitness goal: {predicted_goal}")
    health_goals = predicted_goal.lower().replace(' ', '_')
    goal = calculate_nutrition_targets(weight, age, gender, activity_level, health_goals, height)
    print(f"\nYour personalized daily targets:")
    print(f"Calories: {goal['calories']} kcal | Protein: {goal['protein']}g | Carbs: {goal['carbs']}g | Fat: {goal['fat']}g\n")
    print(f"Macro breakdown: Protein {goal['protein_kcal']} kcal ({goal['protein_kcal']*100//goal['calories']}%), Fat {goal['fat_kcal']} kcal ({goal['fat_kcal']*100//goal['calories']}%), Carbs {goal['carbs_kcal']} kcal ({goal['carbs_kcal']*100//goal['calories']}%)\n")

    # Step 4: Cluster all recipes (for personalization)
    vectorizer = CountVectorizer(token_pattern=r'[^,]+')
    X = vectorizer.fit_transform(recipes['ingredient_str'])
    k = min(6, len(recipes)) if len(recipes) >= 6 else 1
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    recipes['cluster'] = kmeans.fit_predict(X)

    # Step 5: Load upvote history and collect preferred clusters
    upvote_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'last_upvotes.json')
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

    # Step 6: Filter recipes by user filters (ingredients, macros, cuisine, dietary)
    veg_only = (dietary_pref == 'Vegetarian' or dietary_pref == 'Vegan')
    filtered_recipes = recipes.copy()
    if veg_only:
        non_veg_keywords = [
            "chicken", "mutton", "egg", "fish", "prawn", "meat", "lamb", "beef", "pork", "shrimp", "trotter"
        ]
        def is_veg(row):
            name = str(row.get('TranslatedRecipeName', '')).lower()
            ings = str(row.get('Cleaned-Ingredients', '')).lower()
            return not any(k in name or k in ings for k in non_veg_keywords)
        filtered_recipes = filtered_recipes[filtered_recipes.apply(is_veg, axis=1)].copy()
    if preferred_cuisine:
        cuisine_mask = filtered_recipes['Cuisine'].astype(str).str.strip().str.lower() == preferred_cuisine
        cuisine_recipes = filtered_recipes[cuisine_mask]
        if cuisine_recipes.empty:
            print("No recipes found for this cuisine. Showing results from all cuisines instead.")
        else:
            filtered_recipes = cuisine_recipes

    # Jaccard similarity filtering
    user_ings_set = set(user_ingredients)
    def jaccard_sim(set1, set2):
        set1, set2 = set(set1), set(set2)
        return len(set1 & set2) / len(set1 | set2) if set1 | set2 else 0.0
    filtered_recipes['jaccard'] = filtered_recipes['ingredient_list'].apply(lambda ings: jaccard_sim(user_ings_set, ings))
    filtered_recipes = filtered_recipes[filtered_recipes['jaccard'] >= 0.2].copy()
    if filtered_recipes.empty:
        print("No recipes found with your available ingredients. Try adding more or different ingredients.")
        return
    filtered_recipes = filtered_recipes.sort_values('jaccard', ascending=False)

    # Step 7: Generate meal plan, prefer but not force upvoted clusters
    meal_types = ['breakfast', 'lunch', 'dinner', 'snack']
    meal_plan = {}
    used_names = set()
    for meal in meal_types:
        # For snack, allow lower calories
        if meal == 'snack':
            candidates = filtered_recipes[(filtered_recipes['EstimatedCalories'] < 300) & (~filtered_recipes['TranslatedRecipeName'].isin(used_names))]
        else:
            candidates = filtered_recipes[(filtered_recipes['EstimatedCalories'] >= 300) & (filtered_recipes['EstimatedCalories'] <= 800) & (~filtered_recipes['TranslatedRecipeName'].isin(used_names))]
        if candidates.empty:
            continue
        # Prefer upvoted clusters if available, but fallback to best match
        if preferred_clusters:
            cluster_candidates = candidates[candidates['cluster'].isin(preferred_clusters)]
            if not cluster_candidates.empty:
                best = cluster_candidates.iloc[0]
            else:
                best = candidates.iloc[0]
        else:
            best = candidates.iloc[0]
        used_names.add(best['TranslatedRecipeName'])
        meal_plan[meal] = best

    # Step 8: Show recipe info for each meal
    print("\nRecommended Meal Plan:")
    for meal in meal_types:
        if meal in meal_plan:
            rec = meal_plan[meal]
            print(f"\n{meal.capitalize()}: {rec['TranslatedRecipeName']}")
            print(f"Calories: {rec['EstimatedCalories']}, Protein: {rec['EstimatedProtein']}g, Carbs: {rec['EstimatedCarbs']}g, Fat: {rec['EstimatedFat']}g")
            print(f"Matching ingredients: {', '.join(sorted(set(user_ings_set) & set(rec['ingredient_list'])))}")
            print(f"Jaccard Similarity: {rec['jaccard']:.2f}")
            print(f"Recipe URL: {rec.get('URL', '')}")

    # Step 9: Upvote-based personalization
    upvote_map = {}
    upvote_prompt = input("\nDo you want to upvote any meals? (breakfast, lunch, dinner, snack — comma-separated or type 'none'): ").strip().lower()
    if upvote_prompt and upvote_prompt != 'none':
        upvote_meals = [m.strip() for m in upvote_prompt.split(',') if m.strip() in meal_types and m.strip() in meal_plan]
        for m in upvote_meals:
            upvote_map[m] = meal_plan[m]['TranslatedRecipeName']
        # Load previous upvotes if file exists
        if os.path.exists(upvote_file):
            try:
                with open(upvote_file, 'r', encoding='utf-8') as f:
                    prev = json.load(f)
                if isinstance(prev, dict):
                    upvote_map = {**prev, **upvote_map}  # update only selected meals
            except Exception:
                pass
        # Save upvote map
        try:
            with open(upvote_file, 'w', encoding='utf-8') as f:
                json.dump(upvote_map, f, ensure_ascii=False, indent=2)
            print("Upvote(s) saved! Your preferences will be used for future recommendations.")
        except Exception:
            print("Could not save upvote.")
    else:
        print("No upvotes recorded.")

if __name__ == "__main__":
    run_recipe_recommender()
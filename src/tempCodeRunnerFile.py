# meal_plan_generator.py

import pandas as pd
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

NON_VEG_KEYWORDS = ["chicken", "fish", "egg", "mutton", "lamb", "prawn", "shrimp", "meat", "beef", "pork", "trotter"]

INDIAN_CUISINES = [
    "indian", "south indian", "north indian", "punjabi", "gujarati", "maharashtrian", "bengali", "rajasthani", "tamil nadu", "andhra", "udupi", "chettinad", "kerala", "hyderabadi", "goan", "mughlai", "sindhi"
]
NON_INDIAN_CUISINES = [
    "mexican", "italian", "continental", "korean", "japanese", "greek", "thai", "chinese", "french", "american", "mediterranean", "spanish", "vietnamese", "lebanese", "turkish", "german", "russian"
]

# --- Ingredient Matching with TF-IDF Cosine Similarity ---
def get_best_recipe_matches(recipes, user_ingredients, veg_only=False, top_n=20, min_similarity=0.2):
    # Fallback for missing 'Cleaned-Ingredients' and 'Cuisine'
    if 'Cleaned-Ingredients' not in recipes.columns:
        recipes['Cleaned-Ingredients'] = ''
    if 'Cuisine' not in recipes.columns:
        recipes['Cuisine'] = ''
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

# --- Ingredient Matching with TF-IDF Cosine Similarity ---
def get_best_recipe_matches(recipes, user_ingredients, veg_only=False, top_n=20, min_similarity=0.2):
    # Fallback for missing 'Cleaned-Ingredients' and 'Cuisine'
    if 'Cleaned-Ingredients' not in recipes.columns:
        recipes['Cleaned-Ingredients'] = ''
    if 'Cuisine' not in recipes.columns:
        recipes['Cuisine'] = ''
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
    result = '. '.join([c if c.endswith(('.', '!', '?')) else c + '.' for c in cleaned])
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
    return " → ".join([s.split('.')[0] for s in summary if s])

def get_meal_tag(meal):
    if meal['protein'] >= 18:
        return "🌿 High Protein"
    if meal['calories'] <= 350:
        return "🔥 Low Calorie"
    if meal['protein'] >= 15 and meal['calories'] >= 400:
        return "💪 Muscle Building"
    return "🧘 Balanced"

def calculate_nutrition_targets(weight, age, gender, activity_level, health_goal, height=170):
    # 1. Calculate BMR
    if gender.lower() == 'm':
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    else:
        bmr = 10 * weight + 6.25 * height - 5 * age - 161
    # 2. Activity factor
    activity_factors = {'sedentary': 1.2, 'moderate': 1.55, 'active': 1.725}
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

def is_valid_meal(meal):
    # Exclude meals with 0 protein or 0 carbs or 0 calories
    return meal['protein'] > 0 and meal['carbs'] > 0 and meal['calories'] > 0

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
                'cuisine': rec.get('Cuisine', '')
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

def is_valid_snack(row):
    name = row['TranslatedRecipeName'].lower()
    ingredients = str(row['Cleaned-Ingredients']).lower()
    for k in SNACK_EXCLUDE_KEYWORDS:
        if k in name or k in ingredients:
            return False
    return True

def select_snack(recipes, user_ingredients, used_names, total, goal):
    # Fallback for missing 'Cuisine'
    if 'Cuisine' not in recipes.columns:
        recipes['Cuisine'] = ''
    snack_cuisine = ["gujarati", "maharashtrian", "south indian", "tamil nadu", "kerala", "andhra", "udupi"]
    snack_candidates = recipes[
        (recipes['EstimatedCalories'] < 250) &
        (~recipes['TranslatedRecipeName'].isin(used_names))
    ]
    snack_candidates = snack_candidates[
        snack_candidates['Cleaned-Ingredients'].str.contains('|'.join(user_ingredients), case=False)
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

# --- Explain recipe recommendation ---
def explain_recommendation(recipe, user_ingredients):
    recipe_ings = set(str(recipe.get('Cleaned-Ingredients', '')).lower().split(','))
    user_ings = set([i.strip().lower() for i in user_ingredients])
    overlap = recipe_ings & user_ings
    percent = int(100 * len(overlap) / max(1, len(recipe_ings)))
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
    return f"{reason}. {macro}."

def main():
    filepath = 'c:\\Users\\lalwa\\Desktop\\AIML Lab SEE\\indian_recipes_enriched_with_per_person_nutrition.csv'
    recipes = load_recipes(filepath)

    user_ingredients = [i.strip().lower() for i in input("Enter your available ingredients (comma-separated): ").split(',')]
    weight = float(input("Enter your weight (kg): "))
    age = int(input("Enter your age: "))
    gender = input("Enter your gender (M/F): ").strip().lower()
    activity_level = input("Enter your activity level (sedentary/moderate/active): ").strip().lower()
    health_goals = input("Enter your health goals (weight_loss/muscle_gain/maintenance): ").strip().lower()
    # Removed vegetarian input; recommend both veg and non-veg
    height_in = input("Enter your height in cm (or press Enter for 170): ").strip()
    height = float(height_in) if height_in else 170

    goal = calculate_nutrition_targets(weight, age, gender, activity_level, health_goals, height)
    print(f"\nYour personalized daily targets:")
    print(f"Calories: {goal['calories']} kcal | Protein: {goal['protein']}g | Carbs: {goal['carbs']}g | Fat: {goal['fat']}g\n")
    print(f"Macro breakdown: Protein {goal['protein_kcal']} kcal ({goal['protein_kcal']*100//goal['calories']}%), Fat {goal['fat_kcal']} kcal ({goal['fat_kcal']*100//goal['calories']}%), Carbs {goal['carbs_kcal']} kcal ({goal['carbs_kcal']*100//goal['calories']}%)\n")

    # --- TF-IDF ingredient matching and veg filter ---
    filtered_recipes = get_best_recipe_matches(recipes, user_ingredients, veg_only=False, top_n=30, min_similarity=0.2)
    if filtered_recipes.empty:
        print("⚠️ Not enough ingredients for a complete meal plan.")
        return

    meal_plan = generate_meal_plan(filtered_recipes, user_ingredients, health_goals, goal)

    total = {"calories": 0, "protein": 0, "carbs": 0, "fat": 0}
    used_names = []
    staple_summary = {}
    staple_log = set()
    for meal in meal_plan:
        used_names.append(meal['recipe_name'])
        tag = get_meal_tag(meal)
        summary = get_instruction_summary(meal['instructions'])
        reason = explain_recommendation(meal, user_ingredients)
        staple_note = staple_summary.get(meal['meal'], "")
        # Prevent excessive staple logs
        if staple_note and (meal['meal'], staple_note) not in staple_log:
            print(f"🍽️ Added {staple_note} to {meal['meal']}.")
            staple_log.add((meal['meal'], staple_note))
        print(f"{meal['meal']}: {meal['recipe_name']} {tag} {staple_note}")
        print(f"Serving: 1")
        print(f"Summary: {summary}")
        print(f"Why: {reason}")
        print(f"Calories: {meal['calories']}, Protein: {meal['protein']}g, Carbs: {meal['carbs']}g, Fat: {meal['fat']}g")
        print(f"Recipe URL: {meal['url']}\n")
        total["calories"] += meal["calories"]
        total["protein"] += meal["protein"]
        total["carbs"] += meal["carbs"]
        total["fat"] += meal["fat"]

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

    # Optionally offer user to see more suggestions
    more = input("\nWant suggestions to boost your meal? (y/n): ").strip().lower()
    if more == 'y':
        print("Try: 1 glass milk (120 kcal, 6g protein), 1 banana (100 kcal), 50g paneer (130 kcal, 8g protein), 2 tbsp peanuts (100 kcal, 5g protein)")

if __name__ == "__main__":
    main()
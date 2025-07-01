import os
import pandas as pd
from data_loader import load_recipes
from ingredient_utils import normalize_ingredient, get_best_recipe_matches
from nutrition_utils import calculate_nutrition_targets
from meal_planner import generate_meal_plan
from recommendation_engine import cluster_recipes, load_upvote_history, save_upvote_history
from shopping_list import generate_shopping_list
from fitness_goal_classifier import train_fitness_goal_classifier
import json

def run_recipe_recommender():
    filepath = os.path.join(os.path.dirname(__file__), '..', 'indian_recipes_enriched_with_per_person_nutrition.csv')
    recipes = load_recipes(filepath)
    recipes = recipes.dropna(subset=['Cleaned-Ingredients'])

    # User input
    user_ingredients = [normalize_ingredient(i) for i in input("Enter your available ingredients (comma-separated): ").split(',') if i.strip()]
    weight = float(input("Enter your weight (kg): "))
    age = int(input("Enter your age: "))
    gender_input = input("Enter your gender (male/female): ").strip().lower()
    gender_map = {'m': 'Male', 'male': 'Male', 'f': 'Female', 'female': 'Female'}
    gender = gender_map.get(gender_input)
    if gender is None:
        print("Invalid gender. Please enter 'male' or 'female'.")
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
    dietary_pref_input = input("Enter your dietary preference (Omnivore/Vegetarian): ").strip().lower()
    dietary_map = {'omnivore': 'Omnivore', 'vegetarian': 'Vegetarian'}
    dietary_pref = dietary_map.get(dietary_pref_input)
    if dietary_pref is None:
        print("Invalid dietary preference. Please enter Omnivore or Vegetarian.")
        return
    height_in = input("Enter your height in cm (or press Enter for 170): ").strip()
    height = float(height_in) if height_in else 170
    preferred_cuisine = input("""
Choose a preferred Indian cuisine (optional):
Options: North Indian, South Indian, Gujarati, Punjabi, Bengali, Maharashtrian, Rajasthani, Kashmiri, Chettinad, Andhra, Tamil Nadu, Kerala, Bihari, Assamese, Hyderabadi, Lucknowi, Goan, Sindhi, Coorg, Malvani, Mangalorean, Oriya Recipes, North Karnataka, South Karnataka, Karnataka, Awadhi, Konkan, Kongunadu, Uttarakhand-North Kumaon, Jharkhand, Nagaland
Type one of the above or press Enter to skip:
""").strip().lower()

    # Fitness goal prediction
    dataset_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'nutrition_dataset.csv')
    clf, encoders, le_goal = train_fitness_goal_classifier(dataset_path)
    # Use hybrid prediction (Naive Bayes + rules)
    from fitness_goal_classifier import predict_fitness_goal_hybrid
    predicted_goal = predict_fitness_goal_hybrid(
        clf, encoders, le_goal, age, gender, height, weight, activity_level, dietary_pref
    )
    print(f"Predicted fitness goal: {predicted_goal}")
    health_goals = predicted_goal.lower().replace(' ', '_')
    goal = calculate_nutrition_targets(weight, age, gender, activity_level, health_goals, height)
    print(f"\nYour personalized daily targets:")
    print(f"Calories: {goal['calories']} kcal | Protein: {goal['protein']}g | Carbs: {goal['carbs']}g | Fat: {goal['fat']}g\n")
    print(f"Macro breakdown: Protein {goal['protein_kcal']} kcal ({goal['protein_kcal']*100//goal['calories']}%), Fat {goal['fat_kcal']} kcal ({goal['fat_kcal']*100//goal['calories']}%), Carbs {goal['carbs_kcal']} kcal ({goal['carbs_kcal']*100//goal['calories']}%)\n")

    # Clustering and upvote personalization
    recipes = cluster_recipes(recipes)
    upvote_file = os.path.join(os.path.dirname(__file__), 'last_upvotes.json')
    preferred_clusters = load_upvote_history(recipes, upvote_file)

    # Filtering
    veg_only = (dietary_pref == 'Vegetarian' or dietary_pref == 'Vegan')
    filtered_recipes = recipes.copy()
    if veg_only:
        from constants import NON_VEG_KEYWORDS
        def is_veg(row):
            name = str(row.get('TranslatedRecipeName', '')).lower()
            ings = str(row.get('Cleaned-Ingredients', '')).lower()
            return not any(k in name or k in ings for k in NON_VEG_KEYWORDS)
        filtered_recipes = filtered_recipes[filtered_recipes.apply(is_veg, axis=1)].copy()
    # Soft cuisine preference: prioritize preferred cuisine, but don't filter out others
    if preferred_cuisine:
        cuisine_mask = filtered_recipes['Cuisine'].astype(str).str.strip().str.lower() == preferred_cuisine
        cuisine_recipes = filtered_recipes[cuisine_mask]
        other_recipes = filtered_recipes[~cuisine_mask]
        # Take as many as possible from preferred cuisine, then fill with others
        filtered_recipes = pd.concat([cuisine_recipes, other_recipes], ignore_index=True)
    # Load last recommended recipes
    last_recommended_file = os.path.join(os.path.dirname(__file__), 'last_recommended.json')
    last_recommended = set()
    if os.path.exists(last_recommended_file):
        try:
            with open(last_recommended_file, 'r', encoding='utf-8') as f:
                last_recommended = set(json.load(f))
        except Exception:
            last_recommended = set()
    filtered_recipes = get_best_recipe_matches(filtered_recipes, user_ingredients, veg_only=veg_only, top_n=30, min_similarity=0.2)
    # Remove last recommended recipes if possible
    if not filtered_recipes.empty and last_recommended:
        filtered_recipes = filtered_recipes[~filtered_recipes['TranslatedRecipeName'].isin(last_recommended)]
        if filtered_recipes.empty:
            # If all are filtered out, allow previous recipes (so user is not left with nothing)
            filtered_recipes = get_best_recipe_matches(filtered_recipes, user_ingredients, veg_only=veg_only, top_n=30, min_similarity=0.2)

    if filtered_recipes.empty:
        print("⚠️ Not enough ingredients for a complete meal plan.")
        # Fallback: recommend 4 diverse recipes (one per major cuisine cluster or at random)
        print("No exact ingredient matches found. Showing fallback recommendations.")
        # Try to get 4 recipes from different cuisines
        fallback_meals = []
        cuisines_seen = set()
        for _, row in recipes.iterrows():
            cuisine = str(row.get('Cuisine', '')).strip().lower()
            if cuisine not in cuisines_seen and len(fallback_meals) < 4:
                fallback_meals.append(row)
                cuisines_seen.add(cuisine)
            if len(fallback_meals) == 4:
                break
        # If not enough cuisines, fill with random recipes
        if len(fallback_meals) < 4:
            needed = 4 - len(fallback_meals)
            others = recipes.loc[~recipes.index.isin([r.name for r in fallback_meals])]
            fallback_meals += list(others.sample(n=needed, random_state=42).itertuples(index=False))
        # Build meal_plan-like structure
        meal_types = ['Breakfast', 'Lunch', 'Snack', 'Dinner']
        meal_plan = []
        for meal, row in zip(meal_types, fallback_meals):
            def clean_name(name):
                import re
                name = re.sub(r'<.*?>', '', str(name))
                name = re.sub(r'[^\w\s\-\(\)]', '', name)
                return name.strip()
            meal_plan.append({
                'meal': meal,
                'recipe_name': clean_name(getattr(row, 'TranslatedRecipeName', getattr(row, 'name', 'Recipe'))),
                'calories': getattr(row, 'Calories', 0),
                'protein': getattr(row, 'Protein_g', 0),
                'carbs': getattr(row, 'Carbs_g', 0),
                'fat': getattr(row, 'Fat_g', 0),
                'ingredients': getattr(row, 'Ingredients', ''),
                'cleaned_ingredients': getattr(row, 'Cleaned-Ingredients', ''),
                'instructions': getattr(row, 'TranslatedInstructions', '')
            })
        print("\nRecommended Meal Plan:")
        for meal in meal_plan:
            print(f"{meal['meal']}: {meal['recipe_name']} | Calories: {meal['calories']} | Protein: {meal['protein']}g | Carbs: {meal['carbs']}g | Fat: {meal['fat']}g")
        shopping_list = generate_shopping_list(meal_plan, user_ingredients)
        print("\n🛒 Shopping List (Ingredients you need to buy):")
        if shopping_list:
            for item in shopping_list:
                print("-", item)
        else:
            print("You have all the ingredients needed for your meal plan!")
        return

    # Meal plan
    meal_plan = generate_meal_plan(filtered_recipes, user_ingredients, health_goals, goal)
    if not meal_plan:
        return
    # Save recommended recipes for next time
    try:
        with open(last_recommended_file, 'w', encoding='utf-8') as f:
            json.dump([meal['recipe_name'] for meal in meal_plan], f, ensure_ascii=False, indent=2)
    except Exception:
        pass
    print("\nRecommended Meal Plan:")
    for meal in meal_plan:
        print(f"{meal['meal']}: {meal['recipe_name']} | Calories: {meal['calories']} | Protein: {meal['protein']}g | Carbs: {meal['carbs']}g | Fat: {meal['fat']}g")

    # Nutrition summary after meal plan
    total_calories = sum(meal['calories'] for meal in meal_plan)
    total_protein = sum(meal['protein'] for meal in meal_plan)
    total_carbs = sum(meal['carbs'] for meal in meal_plan)
    total_fat = sum(meal['fat'] for meal in meal_plan)
    print("\nNutritional Summary for Your Meal Plan:")
    print(f"Total Calories: {total_calories} kcal / Target: {goal['calories']} kcal")
    print(f"Total Protein: {total_protein}g / Target: {goal['protein']}g")
    print(f"Total Carbs: {total_carbs}g / Target: {goal['carbs']}g")
    print(f"Total Fat: {total_fat}g / Target: {goal['fat']}g\n")
    # Suggestions if under target
    suggestions = []
    if total_protein < goal['protein']:
        suggestions.append("Add a boiled egg, a cup of curd, or a handful of roasted chana for extra protein.")
    if total_carbs < goal['carbs']:
        suggestions.append("Have a banana, a roti, or a bowl of rice to increase carbs.")
    if total_fat < goal['fat']:
        suggestions.append("Add a spoon of ghee, a few nuts, or a slice of cheese for healthy fats.")
    if total_calories < goal['calories']:
        suggestions.append("Consider a glass of milk, a fruit, or a small snack to meet your calorie goal.")
    if suggestions:
        print("Suggestions to meet your nutritional goals:")
        for s in suggestions:
            print("-", s)
    else:
        print("Your meal plan meets or exceeds all your nutritional goals!")

    # Shopping list
    shopping_list = generate_shopping_list(meal_plan, user_ingredients)
    print("\n🛒 Shopping List (Ingredients you need to buy):")
    if shopping_list:
        for item in shopping_list:
            print("-", item)
    else:
        print("You have all the ingredients needed for your meal plan!")

    # Upvote logic
    meal_types = [m['meal'].lower() for m in meal_plan]
    upvote_map = {}
    upvote_prompt = input("\nDo you want to upvote any meals? (breakfast, lunch, dinner, snack — comma-separated or type 'none'): ").strip().lower()
    if upvote_prompt and upvote_prompt != 'none':
        upvote_meals = [m.strip() for m in upvote_prompt.split(',') if m.strip() in meal_types]
        for m in upvote_meals:
            for meal in meal_plan:
                if meal['meal'].lower() == m:
                    upvote_map[m] = meal['recipe_name']
        if os.path.exists(upvote_file):
            try:
                with open(upvote_file, 'r', encoding='utf-8') as f:
                    prev = json.load(f)
                if isinstance(prev, dict):
                    upvote_map = {**prev, **upvote_map}
            except Exception:
                pass
        if save_upvote_history(upvote_map, upvote_file):
            print("Upvote(s) saved! Your preferences will be used for future recommendations.")
        else:
            print("Could not save upvote.")
    else:
        print("No upvotes recorded.")

if __name__ == "__main__":
    run_recipe_recommender()
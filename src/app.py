import streamlit as st
import os
import json
import pandas as pd
from data_loader import load_recipes
from ingredient_utils import normalize_ingredient, get_best_recipe_matches
from nutrition_utils import calculate_nutrition_targets
from meal_planner import generate_meal_plan
from recommendation_engine import cluster_recipes, load_upvote_history, save_upvote_history, get_clusters_for_recipes, generate_meal_plan_with_upvotes
from shopping_list import generate_shopping_list
from fitness_goal_classifier import train_fitness_goal_classifier, predict_fitness_goal_hybrid

def get_file_path(filename):
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, filename)

# --- Ingredient List Management ---
def load_user_ingredients():
    path = get_file_path('user_ingredients.json')
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return []

def save_user_ingredients(ingredients):
    path = get_file_path('user_ingredients.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(ingredients, f, ensure_ascii=False, indent=2)

# --- User Profile Management ---
def get_user_profile():
    st.subheader('User Details')
    col1, col2, col3 = st.columns(3)
    weight = col1.text_input('Weight (kg)', value='70')
    age = col2.text_input('Age', value='30')
    height = col3.text_input('Height (cm)', value='170')
    # Validation for numeric fields
    weight_error = age_error = height_error = ''
    try:
        weight_val = float(weight)
        if weight_val <= 0:
            weight_error = 'Weight must be a positive number.'
    except Exception:
        weight_error = 'Weight must be a number.'
    try:
        age_val = int(age)
        if age_val <= 0:
            age_error = 'Age must be a positive integer.'
    except Exception:
        age_error = 'Age must be a number.'
    try:
        height_val = float(height)
        if height_val <= 0:
            height_error = 'Height must be a positive number.'
    except Exception:
        height_error = 'Height must be a number.'
    if weight_error:
        col1.error(weight_error)
    if age_error:
        col2.error(age_error)
    if height_error:
        col3.error(height_error)
    gender = st.radio('Gender', ['Male', 'Female'], horizontal=True)
    activity_level = st.selectbox('Activity Level', ['Sedentary', 'Lightly Active', 'Moderately Active', 'Very Active'])
    # Change order and label for dietary preference
    dietary_options = ['Vegetarian', 'Non Veg']
    dietary_pref_ui = st.radio('Dietary Preference', dietary_options, horizontal=True, index=0)
    # Map UI label to internal value
    dietary_pref = 'Vegetarian' if dietary_pref_ui == 'Vegetarian' else 'Omnivore'
    cuisines = [
        'North Indian', 'South Indian', 'Gujarati', 'Punjabi', 'Bengali', 'Maharashtrian', 'Rajasthani', 'Kashmiri',
        'Chettinad', 'Andhra', 'Tamil Nadu', 'Kerala', 'Bihari', 'Assamese', 'Hyderabadi', 'Lucknowi', 'Goan',
        'Sindhi', 'Coorg', 'Malvani', 'Mangalorean', 'Oriya Recipes', 'North Karnataka', 'South Karnataka',
        'Karnataka', 'Awadhi', 'Konkan', 'Kongunadu', 'Uttarakhand-North Kumaon', 'Jharkhand', 'Nagaland'
    ]
    preferred_cuisines = st.multiselect('Preferred Cuisines (optional)', cuisines)
    # If any error, return None to indicate invalid input
    if weight_error or age_error or height_error:
        return None
    return {
        'weight': weight_val,
        'age': age_val,
        'height': height_val,
        'gender': gender,
        'activity_level': activity_level,
        'dietary_pref': dietary_pref,
        'preferred_cuisines': [c.lower() for c in preferred_cuisines]
    }

# --- Fitness Goal Prediction ---
def get_fitness_goal_and_targets(user_profile):
    dataset_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'nutrition_dataset.csv')
    clf, encoders, le_goal = train_fitness_goal_classifier(dataset_path)
    goal = predict_fitness_goal_hybrid(
        clf, encoders, le_goal,
        user_profile['age'], user_profile['gender'], user_profile['height'],
        user_profile['weight'], user_profile['activity_level'], user_profile['dietary_pref']
    )
    health_goals = goal.lower().replace(' ', '_')
    targets = calculate_nutrition_targets(
        user_profile['weight'], user_profile['age'], user_profile['gender'],
        user_profile['activity_level'], health_goals, user_profile['height']
    )
    return goal, targets

# --- Meal Plan Recommendation ---
def recommend_meal_plan(user_profile, user_ingredients):
    import pandas as pd
    from recommendation_engine import get_clusters_for_recipes, generate_meal_plan_with_upvotes
    recipes_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'indian_recipes_enriched_with_per_person_nutrition.csv')
    recipes = load_recipes(recipes_path)
    recipes = recipes.dropna(subset=['Cleaned-Ingredients'])
    recipes = cluster_recipes(recipes)
    # Remove last recommended
    last_recommended_path = get_file_path('last_recommended.json')
    last_recommended = set()
    if os.path.exists(last_recommended_path):
        try:
            with open(last_recommended_path, 'r', encoding='utf-8') as f:
                last_recommended = set(json.load(f).values())
        except Exception:
            last_recommended = set()
    # Ingredient filtering
    user_ings = [normalize_ingredient(i['ingredient']) for i in user_ingredients if not i.get('used', False)]
    veg_only = (user_profile['dietary_pref'] == 'Vegetarian')
    filtered = recipes.copy()
    if veg_only:
        from constants import NON_VEG_KEYWORDS
        def is_veg(row):
            name = str(row.get('TranslatedRecipeName', '')).lower()
            ings = str(row.get('Cleaned-Ingredients', '')).lower()
            return not any(k in name or k in ings for k in NON_VEG_KEYWORDS)
        filtered = filtered[filtered.apply(is_veg, axis=1)].copy()
    # Cuisine soft preference
    if user_profile['preferred_cuisines']:
        mask = filtered['Cuisine'].astype(str).str.strip().str.lower().isin(user_profile['preferred_cuisines'])
        cuisine_recipes = filtered[mask]
        other_recipes = filtered[~mask]
        filtered = pd.concat([cuisine_recipes, other_recipes], ignore_index=True)
    filtered = get_best_recipe_matches(filtered, user_ings, veg_only=veg_only, top_n=30, min_similarity=0.2)
    # --- Upvote-based personalization ---
    upvotes_path = get_file_path('last_upvotes.json')
    upvoted_recipes = []
    if os.path.exists(upvotes_path):
        try:
            with open(upvotes_path, 'r', encoding='utf-8') as f:
                upvoted_recipes = list(json.load(f).values())
        except Exception:
            upvoted_recipes = []
    upvoted_clusters = get_clusters_for_recipes(recipes, upvoted_recipes) if upvoted_recipes else set()
    # Generate meal plan with upvote clusters if available
    health_goals = user_profile.get('fitness_goal', '').lower().replace(' ', '_')
    targets = user_profile.get('nutrition_targets', {})
    if not targets:
        _, targets = get_fitness_goal_and_targets(user_profile)
    if upvoted_clusters:
        meal_plan = generate_meal_plan_with_upvotes(filtered, user_ings, health_goals, targets, upvoted_clusters, last_recommended)
        used_upvotes = True
    else:
        meal_plan = generate_meal_plan(filtered, user_ings, health_goals, targets)
        used_upvotes = False
    # Save new recommendations
    if meal_plan:
        to_save = {m['meal'].lower(): m['recipe_name'] for m in meal_plan}
        with open(last_recommended_path, 'w', encoding='utf-8') as f:
            json.dump(to_save, f, ensure_ascii=False, indent=2)
    return meal_plan, used_upvotes

# --- Shopping List ---
def get_shopping_list(meal_plan, user_ingredients):
    user_ings = [normalize_ingredient(i['ingredient']) for i in user_ingredients if not i.get('used', False)]
    # Collect all ingredients needed for the meal plan
    needed_ingredients = set()
    for meal in meal_plan:
        # Find the recipe in the dataset to get its ingredients
        recipes_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'indian_recipes_enriched_with_per_person_nutrition.csv')
        recipes_df = pd.read_csv(recipes_path)
        match = recipes_df[recipes_df['TranslatedRecipeName'].str.lower() == meal['recipe_name'].strip().lower()]
        if not match.empty:
            ings = str(match.iloc[0]['Cleaned-Ingredients']).split(',')
            needed_ingredients.update([normalize_ingredient(i) for i in ings if i.strip()])
    # Find missing ingredients
    missing = [ing for ing in needed_ingredients if ing not in user_ings]
    return missing

# --- Main Streamlit App ---
st.set_page_config(page_title="Swaad.ai – Personalized Indian Meals for Your Health Goals", layout="wide")
st.title("Swaad.ai – Personalized Indian Meals for Your Health Goals")
st.divider()

# 1. Ingredients Input
st.header('🧂 Ingredients')
ing_list = load_user_ingredients()
with st.form('add_ingredient'):
    col1, col2 = st.columns([2,1])
    new_ing = col1.text_input('Ingredient name')
    qty = col2.text_input('Quantity')  # No 'optional' label
    add_btn = st.form_submit_button('Add')
    if add_btn and new_ing:
        ing_obj = {'ingredient': new_ing.strip().lower(), 'quantity': qty.strip(), 'used': False}
        ing_list.append(ing_obj)
        save_user_ingredients(ing_list)

# --- Ingredient List Display with Edit/Finished ---
max_visible = 5
if len(ing_list) <= max_visible:
    visible_ings = ing_list
    extra_ings = []
else:
    visible_ings = ing_list[:max_visible]
    extra_ings = ing_list[max_visible:]

# Track which ingredient is being edited
if 'edit_idx' not in st.session_state:
    st.session_state['edit_idx'] = None

def ingredient_row(idx, ing):
    cols = st.columns([3,2,1,1])
    cols[0].write(ing['ingredient'])
    # Edit mode for quantity
    if st.session_state['edit_idx'] == idx:
        new_qty = cols[1].text_input('Quantity', value=ing.get('quantity',''), key=f'edit_qty_{idx}')
        if cols[2].button('Save', key=f'save_{idx}'):
            ing['quantity'] = new_qty
            save_user_ingredients(ing_list)
            st.session_state['edit_idx'] = None
        if cols[3].button('Cancel', key=f'cancel_{idx}'):
            st.session_state['edit_idx'] = None
    else:
        cols[1].write(ing.get('quantity',''))
        if cols[2].button('Edit', key=f'edit_{idx}'):
            st.session_state['edit_idx'] = idx
        if cols[3].button('Finished', key=f'fin_{idx}'):
            ing_list.pop(idx)
            save_user_ingredients(ing_list)
            # st.experimental_rerun()  # Removed, not needed

st.subheader('Your Ingredients')
for idx, ing in enumerate(visible_ings):
    ingredient_row(idx, ing)
if extra_ings:
    with st.expander(f"Show {len(extra_ings)} more ingredients"):
        for j, ing in enumerate(extra_ings, start=max_visible):
            ingredient_row(j, ing)
st.divider()

# 2. User Details Input
user_profile = get_user_profile()
st.divider()

# --- Meal Plan and Upvote State Management ---
if 'meal_plan' not in st.session_state:
    st.session_state['meal_plan'] = None
if 'meal_targets' not in st.session_state:
    st.session_state['meal_targets'] = None
if 'fitness_goal' not in st.session_state:
    st.session_state['fitness_goal'] = None
if 'upvotes' not in st.session_state or not isinstance(st.session_state['upvotes'], dict):
    st.session_state['upvotes'] = {}

# 3. Recommend Meal Plan Button
if st.button('Recommend Meal Plan'):
    if user_profile is None:
        st.error('Please enter valid numeric values for Weight, Age, and Height.')
    else:
        # Predict fitness goal and targets
        fitness_goal, targets = get_fitness_goal_and_targets(user_profile)
        user_profile['fitness_goal'] = fitness_goal
        user_profile['nutrition_targets'] = targets
        meal_plan, used_upvotes = recommend_meal_plan(user_profile, ing_list)
        st.session_state['meal_plan'] = meal_plan
        st.session_state['meal_targets'] = targets
        st.session_state['fitness_goal'] = fitness_goal
        st.session_state['used_upvotes'] = used_upvotes
        # Reset upvotes for new plan
        st.session_state['upvotes'] = {}
        if meal_plan:
            for meal in meal_plan:
                meal_key = meal['meal'].lower()
                st.session_state['upvotes'][meal_key] = False
        if used_upvotes:
            st.info('Your meal plan has been personalized based on your upvoted recipes!')

# --- Display Meal Plan and Upvotes if available ---
meal_plan = st.session_state.get('meal_plan')
targets = st.session_state.get('meal_targets')
fitness_goal = st.session_state.get('fitness_goal')
used_upvotes = st.session_state.get('used_upvotes', False)
if meal_plan is not None:
    if used_upvotes:
        st.info("Based on your previous upvotes, we’ve prioritized similar meals for you.")
    # --- Nutritional Summary (no emojis, just numbers) ---
    total_cals = sum(m['calories'] for m in meal_plan)
    total_prot = sum(m['protein'] for m in meal_plan)
    total_carb = sum(m['carbs'] for m in meal_plan)
    total_fat = sum(m['fat'] for m in meal_plan)
    st.subheader('Nutritional Summary')
    st.markdown(f"**Total Calories:** {total_cals} kcal / Target: {targets['calories']} kcal")
    st.markdown(f"**Total Protein:** {total_prot}g / Target: {targets['protein']}g")
    st.markdown(f"**Total Carbs:** {total_carb}g / Target: {targets['carbs']}g")
    st.markdown(f"**Total Fat:** {total_fat}g / Target: {targets['fat']}g")
    # --- Smart goal-based completion: add real foods to meet gap ---
    # Define nutritional values for common foods
    food_macros = {
        'roti': {'calories': 100, 'protein': 3, 'carbs': 20, 'fat': 0.5, 'max': 4},
        'curd': {'calories': 60, 'protein': 4, 'carbs': 6, 'fat': 4, 'max': 2},
        'boiled egg': {'calories': 70, 'protein': 6, 'carbs': 1, 'fat': 5, 'max': 3},
        'banana': {'calories': 100, 'protein': 1, 'carbs': 27, 'fat': 0.3, 'max': 2},
        'peanut butter toast': {'calories': 180, 'protein': 7, 'carbs': 18, 'fat': 10, 'max': 2},
        'milk': {'calories': 120, 'protein': 6, 'carbs': 12, 'fat': 4, 'max': 2},
        'fruit bowl': {'calories': 80, 'protein': 1, 'carbs': 20, 'fat': 0.5, 'max': 2},
        'nuts': {'calories': 100, 'protein': 3, 'carbs': 4, 'fat': 9, 'max': 1},
        'sprouts': {'calories': 50, 'protein': 4, 'carbs': 8, 'fat': 0.5, 'max': 1},
        'ghee': {'calories': 45, 'protein': 0, 'carbs': 0, 'fat': 5, 'max': 1},
        'rice': {'calories': 200, 'protein': 4, 'carbs': 45, 'fat': 0.5, 'max': 2},
        'paratha': {'calories': 180, 'protein': 4, 'carbs': 30, 'fat': 6, 'max': 2},
        'dal': {'calories': 140, 'protein': 8, 'carbs': 20, 'fat': 2, 'max': 2},
        'chana': {'calories': 120, 'protein': 7, 'carbs': 20, 'fat': 2, 'max': 1},
    }
    # Compute gaps
    gap_cals = max(0, targets['calories'] - total_cals)
    gap_prot = max(0, targets['protein'] - total_prot)
    gap_carb = max(0, targets['carbs'] - total_carb)
    gap_fat = max(0, targets['fat'] - total_fat)
    # Build a completion plan
    completion = {}
    remain_cals, remain_prot, remain_carb, remain_fat = gap_cals, gap_prot, gap_carb, gap_fat
    is_veg = user_profile['dietary_pref'].lower() == 'vegetarian'
    # List of foods to try, in order of priority, with veg/nonveg filter
    food_order = [
        ('roti', True),
        ('rice', True),
        ('milk', True),
        ('curd', True),
        ('dal', True),
        ('chana', True),
        ('banana', True),
        ('paratha', True),
        ('sprouts', True),
        ('fruit bowl', True),
        ('nuts', True),
        ('peanut butter toast', True),
        ('ghee', True),
        ('boiled egg', False),  # Only if not veg
    ]
    # Add foods in realistic amounts to fill the gap
    for food, veg_ok in food_order:
        if not veg_ok and is_veg:
            continue
        macros = food_macros[food]
        max_qty = macros.get('max', 2)
        qty = 0
        while qty < max_qty and (remain_cals > 0 or remain_prot > 0 or remain_carb > 0 or remain_fat > 0):
            # Only add if it helps fill a gap
            if (remain_prot > 2 and macros['protein'] > 2) or (remain_cals > 80 and macros['calories'] > 50) or (remain_carb > 10 and macros['carbs'] > 10) or (remain_fat > 2 and macros['fat'] > 2):
                completion[food] = completion.get(food, 0) + 1
                remain_prot -= macros['protein']
                remain_cals -= macros['calories']
                remain_carb -= macros['carbs']
                remain_fat -= macros['fat']
                qty += 1
            else:
                break
    if completion:
        add_str = ', '.join([f"{v} x {k}" for k, v in completion.items()])
        st.info('To meet your nutritional goals, add: ' + add_str)
    else:
        st.success('Your meal plan meets or exceeds all your nutritional goals!')
    st.divider()
    # Show meal plan
    st.subheader('Upvote your favorite meals:')
    # --- Robust Upvote Form ---
    if 'upvotes' not in st.session_state or not isinstance(st.session_state['upvotes'], dict):
        st.session_state['upvotes'] = {}
    # Temporary dict to hold form state
    upvote_form_state = {}
    with st.form('upvote_form'):
        for meal in meal_plan:
            meal_key = meal['meal'].lower()
            upvote_key = f"upvote_{meal_key}"
            # Use form state or session state as default
            default_val = st.session_state['upvotes'].get(meal_key, False)
            recipe_url = meal.get('url') or meal.get('RecipeUrl') or meal.get('Recipe URL') or ''
            cols = st.columns([3,2,2,2,2,1])
            if recipe_url:
                cols[0].markdown(f"**{meal['meal']}**: [{meal['recipe_name']}]({recipe_url})")
            else:
                cols[0].markdown(f"**{meal['meal']}**: {meal['recipe_name']}")
            cols[1].write(f"Calories: {meal['calories']}")
            cols[2].write(f"Protein: {meal['protein']}g")
            cols[3].write(f"Carbs: {meal['carbs']}g")
            cols[4].write(f"Fat: {meal['fat']}g")
            upvote_form_state[meal_key] = cols[5].checkbox('Upvote', value=default_val, key=upvote_key)
            st.caption(meal.get('instructions', ''))
        save_upvotes_btn = st.form_submit_button('Save Upvotes')
        if save_upvotes_btn:
            # Only update session state and persist on submit
            st.session_state['upvotes'] = upvote_form_state.copy()
            upvotes_path = get_file_path('last_upvotes.json')
            upvotes = {meal_key: meal['recipe_name'] for meal_key, upvoted in upvote_form_state.items() if upvoted for meal in meal_plan if meal['meal'].lower() == meal_key}
            with open(upvotes_path, 'w', encoding='utf-8') as f:
                json.dump(upvotes, f, ensure_ascii=False, indent=2)
            st.success('Upvotes saved! Your preferences will be used for future recommendations.')
    # Shopping list
    shopping_list = get_shopping_list(meal_plan, ing_list)
    if not shopping_list:
        st.success('✅ You have all the ingredients needed!')
    else:
        with st.expander('🛒 Shopping List'):
            for item in shopping_list:
                st.write('-', item)

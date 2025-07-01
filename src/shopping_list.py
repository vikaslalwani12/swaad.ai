# shopping_list.py
from constants import COMMON_ESSENTIALS
from ingredient_utils import normalize_ingredient

def generate_shopping_list(selected_meals, user_ingredients):
    all_recipe_ings = set()
    for meal in selected_meals:
        ings_str = meal.get('cleaned_ingredients') or meal.get('Cleaned-Ingredients') or meal.get('Ingredients') or meal.get('ingredients') or ''
        for ing in ings_str.split(','):
            ing_clean = normalize_ingredient(ing)
            if ing_clean:
                all_recipe_ings.add(ing_clean)
    all_recipe_ings = set(i for i in all_recipe_ings if i)
    user_ings_set = set(normalize_ingredient(i) for i in user_ingredients if i.strip())
    user_ings_set = set(i for i in user_ings_set if i)
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
    return shopping_list

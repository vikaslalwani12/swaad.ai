# meal_plan_generator.py

import pandas as pd
import random

# Load the dataset
def load_recipes(file_path):
    return pd.read_csv(file_path)

# Filter recipes based on user ingredients
def filter_recipes_by_ingredients(recipes, ingredients):
    filtered_recipes = recipes[recipes['Cleaned-Ingredients'].apply(lambda x: all(item in x for item in ingredients))]
    return filtered_recipes

# Generate a meal plan based on age and health goals
def generate_meal_plan(recipes, age, health_goals):
    meal_plan = {}
    
    # Define meal types
    meal_types = ['Breakfast', 'Lunch', 'Dinner', 'Snack']
    
    # Nutritional considerations based on age and health goals
    if age < 30:
        calorie_limit = 2000
    elif age < 50:
        calorie_limit = 1800
    else:
        calorie_limit = 1600

    if health_goals == 'weight_loss':
        calorie_limit -= 500

    # Select recipes for each meal type
    for meal in meal_types:
        available_recipes = recipes[recipes['EstimatedCalories'] <= calorie_limit]
        if not available_recipes.empty:
            selected_recipe = available_recipes.sample(1).iloc[0]
            meal_plan[meal] = {
                'Recipe Name': selected_recipe['TranslatedRecipeName'],
                'Ingredients': selected_recipe['TranslatedIngredients'],
                'Calories': selected_recipe['EstimatedCalories']
            }
            calorie_limit -= selected_recipe['EstimatedCalories']

    return meal_plan

# Main function to run the meal plan generator
def main():
    file_path = 'c:\\Users\\lalwa\\Desktop\\AIML Lab SEE\\indian_recipes_enriched_with_per_person_nutrition.csv'
    recipes = load_recipes(file_path)

    # User input
    user_ingredients = input("Enter your available ingredients (comma-separated): ").split(',')
    user_age = int(input("Enter your age: "))
    user_health_goals = input("Enter your health goals (weight_loss/maintenance): ")

    # Filter recipes and generate meal plan
    filtered_recipes = filter_recipes_by_ingredients(recipes, [ingredient.strip() for ingredient in user_ingredients])
    meal_plan = generate_meal_plan(filtered_recipes, user_age, user_health_goals)

    # Display the meal plan
    for meal, details in meal_plan.items():
        print(f"{meal}: {details['Recipe Name']} - Ingredients: {details['Ingredients']} - Calories: {details['Calories']}")

if __name__ == "__main__":
    main()
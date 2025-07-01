# data_loader.py
import pandas as pd

def load_recipes(filepath):
    """Load recipes from a CSV file and ensure consistent column names."""
    df = pd.read_csv(filepath)
    df.rename(columns={
        'EstimatedProtein': 'Protein_g',
        'EstimatedCarbs': 'Carbs_g',
        'EstimatedFat': 'Fat_g',
        'EstimatedCalories': 'Calories',
        'TranslatedIngredients': 'Ingredients'
    }, inplace=True)
    return df

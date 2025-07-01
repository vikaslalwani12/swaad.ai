import pandas as pd
import requests
import re
from tqdm import tqdm
import time

# --- CONFIG ---
INPUT_CSV = "Cleaned_Indian_Food_Dataset.csv"
OUTPUT_CSV = "indian_recipes_enriched_with_per_person_nutrition.csv"
GEMINI_API_KEY = "AIzaSyD_vnupJXDFULOLHXT5D7y0SoQ2vkLX1XI"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=" + GEMINI_API_KEY

# --- LOAD DATA ---
df = pd.read_csv(INPUT_CSV)

# --- NUTRITION EXTRACTION ---
def extract_nutrition_from_text(text):
    # Regex to extract nutrition values from Gemini's response
    cal = re.search(r"Calories\s*[:=]\s*([\d.]+)", text, re.IGNORECASE)
    protein = re.search(r"Protein\s*[:=]\s*([\d.]+)", text, re.IGNORECASE)
    carbs = re.search(r"Carbs?\s*[:=]\s*([\d.]+)", text, re.IGNORECASE)
    fat = re.search(r"Fat\s*[:=]\s*([\d.]+)", text, re.IGNORECASE)
    return {
        "EstimatedCalories": float(cal.group(1)) if cal else None,
        "EstimatedProtein": float(protein.group(1)) if protein else None,
        "EstimatedCarbs": float(carbs.group(1)) if carbs else None,
        "EstimatedFat": float(fat.group(1)) if fat else None,
    }

# --- GEMINI CALL ---
def get_nutrition_per_serving(ingredients, cuisine, instructions):
    prompt = f"""
You are a certified nutritionist. Estimate the **nutrition per 1 serving (i.e., for one person)** for the following Indian vegetarian recipe.

Ingredients:
{ingredients}

Cuisine: {cuisine}
Instructions: {instructions}

Return values as a list in this exact format:
Calories: ___ kcal  
Protein: ___ g  
Carbs: ___ g  
Fat: ___ g  
"""
    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }
    try:
        response = requests.post(GEMINI_URL, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        result = response.json()
        text = result["candidates"][0]["content"]["parts"][0]["text"]
        return extract_nutrition_from_text(text)
    except Exception as e:
        print(f"[ERROR] Gemini API call failed: {e}")
        return {"EstimatedCalories": None, "EstimatedProtein": None, "EstimatedCarbs": None, "EstimatedFat": None}

# --- ENRICH DATAFRAME ---
nutrition_results = []
for idx, row in tqdm(df.iterrows(), total=len(df), desc="Enriching recipes with nutrition"):
    try:
        nutrition = get_nutrition_per_serving(
            row.get("Cleaned-Ingredients", ""),
            row.get("Cuisine", ""),
            row.get("TranslatedInstructions", "")
        )
    except Exception as e:
        print(f"[ERROR] Failed to process row {idx}: {e}")
        nutrition = {"EstimatedCalories": None, "EstimatedProtein": None, "EstimatedCarbs": None, "EstimatedFat": None}
    nutrition_results.append(nutrition)
    time.sleep(1)  # Be nice to the API

nutrition_df = pd.DataFrame(nutrition_results)
df = pd.concat([df.reset_index(drop=True), nutrition_df], axis=1)
df["Servings"] = 1

df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Enriched CSV saved as {OUTPUT_CSV}") 
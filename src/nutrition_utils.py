# nutrition_utils.py

def calculate_nutrition_targets(weight, age, gender, activity_level, health_goal, height=170):
    if gender.lower() == 'm' or gender.lower() == 'male':
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    else:
        bmr = 10 * weight + 6.25 * height - 5 * age - 161
    activity_factors = {'sedentary': 1.2, 'lightly active': 1.375, 'moderately active': 1.55, 'very active': 1.725}
    activity_mult = activity_factors.get(activity_level.lower(), 1.2)
    tdee = bmr * activity_mult
    if health_goal == 'muscle_gain':
        protein = 2.0 * weight
    elif health_goal == 'weight_loss':
        protein = 1.6 * weight
    else:
        protein = 1.8 * weight
    protein_kcal = protein * 4
    fat = 0.27 * tdee / 9
    fat_kcal = fat * 9
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

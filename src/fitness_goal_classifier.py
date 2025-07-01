# fitness_goal_classifier.py
# (Move your existing code here from the parent directory)

# ...existing code from your original fitness_goal_classifier.py...

import pandas as pd
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import LabelEncoder

def train_fitness_goal_classifier(csv_path):
    df = pd.read_csv(csv_path)
    # Use the correct column names with spaces
    feature_cols = ['Age', 'Gender', 'Height', 'Weight', 'Activity Level', 'Dietary Preference']
    target_col = 'Fitness Goal'
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    encoders = {}
    for col in feature_cols:
        le = LabelEncoder()
        if X[col].dtype == object:
            X[col] = le.fit_transform(X[col].astype(str))
        else:
            X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le
    le_goal = LabelEncoder()
    y = le_goal.fit_transform(y.astype(str))
    clf = CategoricalNB()
    clf.fit(X, y)
    return clf, encoders, le_goal

def predict_fitness_goal_hybrid(clf, encoders, le_goal, age, gender, height, weight, activity_level, dietary_pref):
    """
    Hybrid fitness goal predictor: Naive Bayes + rule-based overrides for edge cases.
    """
    # 1. Naive Bayes prediction
    input_dict = {
        'Age': age,
        'Gender': gender,
        'Height': height,
        'Weight': weight,
        'Activity Level': activity_level,
        'Dietary Preference': dietary_pref
    }
    input_vec = []
    for col in ['Age', 'Gender', 'Height', 'Weight', 'Activity Level', 'Dietary Preference']:
        val = input_dict[col]
        le = encoders[col]
        if val not in le.classes_:
            val = le.classes_[0]
        input_vec.append(le.transform([str(val)])[0])
    import pandas as pd
    X_df = pd.DataFrame([input_vec], columns=['Age', 'Gender', 'Height', 'Weight', 'Activity Level', 'Dietary Preference'])
    nb_pred = le_goal.inverse_transform([clf.predict(X_df)[0]])[0]

    # 2. Rule-based overrides
    bmi = weight / ((height / 100) ** 2)
    activity = activity_level.strip().lower()
    gender_lc = gender.strip().lower()
    final_pred = nb_pred
    if age > 60:
        final_pred = "Balanced"
    elif bmi < 18.5:
        final_pred = "Weight Gain"
    elif bmi > 25 and activity in ["sedentary", "lightly active"]:
        final_pred = "Weight Loss"
    elif activity == "very active" and weight > 65 and gender_lc == "male":
        final_pred = "Muscle Gain"
    # 4. Otherwise, keep NB prediction
    return final_pred

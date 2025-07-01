# Swaad.ai – Personalized Indian Meal Planner

Swaad.ai is an AI-powered Indian meal planner that helps you discover personalized, healthy, and delicious Indian recipes based on your dietary preferences, fitness goals, and available ingredients. It features a clean Streamlit UI, ingredient tracking, nutrition analysis, and upvote-based personalization.

---

## 🌍 Live Demo

(Coming soon)

---

## 📊 Features

* ✅ **Personalized Meal Recommendations**: Suggests Breakfast, Lunch, Snack, and Dinner based on your ingredients and predicted health goal.
* 🌱 **Vegetarian / Non-Veg / Eggetarian Support**
* 🧬 **Fitness Goal Prediction** using Naive Bayes (Weight Loss / Gain / Muscle Gain)
* 🧪 **Calorie & Macronutrient Tracking**
* 📅 **Dynamic Shopping List Generation**
* ⬆️ **Meal Upvoting for Personalization**
* ⚖️ **Cluster-based Recommendations** with diversity
* 🔄 **Avoids Recently Repeated Meals**
* 🌐 **Cuisine Preference Filtering** (North, South Indian, etc.)
* 📊 **Nutritional Summary & Suggestions to Improve Your Meal Plan**

---

## 🚀 How It Works

1. **Input Ingredients** with quantity (editable UI)
2. **Enter Personal Details**: Age, Weight, Height, Gender, Activity Level, Cuisine, Dietary Preference
3. **Predict Fitness Goal** using a trained Naive Bayes classifier
4. **Generate a Meal Plan** matching your goal & taste
5. **Get Nutritional Breakdown & Suggestions**
6. **Upvote Meals** you like – next time, it recommends similar dishes

---

## 🛁 Installation

```bash
# Clone the repo
git clone https://github.com/your-username/swaad-ai.git
cd swaad-ai

# Install requirements
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## 📓 Dataset & Models

* Recipes Dataset: Curated Indian meals with nutritional metadata
* Clustering: KMeans for recipe grouping
* Goal Prediction: Naive Bayes based on user features
* Similarity: Jaccard similarity on ingredients

---

## 📂 Project Structure

```
├── src
│   ├── app.py                  # Streamlit UI
│   ├── recommendation_engine.py
│   ├── meal_planner.py
│   ├── data_loader.py
│   ├── goal_predictor.py
│   └── utils.py
├── data
│   ├── recipes.csv
│   ├── clusters.pkl
│   └── ...
├── upvotes.json               # Stores meal preferences
├── last_recommended.json      # Avoids recent repeats
├── requirements.txt
└── README.md
```

---

## 🚀 Future Enhancements

* User login / profile-based memory
* Real-time calorie calculator using portion sizes
* Weekly meal planner
* Export to PDF / shareable meal plan
* Recipe image integration

---

## 👨‍💻 Author

Built with ❤️ by \Vikas

---

## ✉️ Contributions

PRs are welcome! Found a bug or want to suggest a feature? [Open an issue](https://github.com/your-username/swaad-ai/issues)!

---

## ✉️ License

This project is licensed under the MIT License.

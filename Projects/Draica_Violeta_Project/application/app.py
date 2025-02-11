from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
import re

#Define paths
MODEL_PATH = r"C:\Users\draic\Desktop\Data Science\Data2\archive\meal_plan_model_random_forest_(tuned).pkl"
DATA_PATH = r"C:\Users\draic\Desktop\Data Science\Data2\archive\healthy_foods_for_meal_planning.csv"
FEATURES_PATH = r"C:\Users\draic\Desktop\Data Science\Data2\archive\feature_names.pkl"

#Load model and data
model = joblib.load(MODEL_PATH)
feature_names = joblib.load(FEATURES_PATH)
data = pd.read_csv(DATA_PATH)

app = Flask(__name__)

#Calculate daily calories
def calculate_daily_calories(weight, height, age, gender, activity_level, goal):
    """Calculate daily caloric needs using Mifflin-St Jeor Equation."""
    if gender.lower() == "male":
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    else:
        bmr = 10 * weight + 6.25 * height - 5 * age - 161

    activity_multipliers = {
        "sedentary": 1.2,
        "lightly active": 1.375,
        "moderately active": 1.55,
        "very active": 1.725,
        "extra active": 1.9,
    }

    daily_calories = bmr * activity_multipliers.get(activity_level.lower(), 1.2)

    if goal == "lose":
        daily_calories *= 0.8
    elif goal == "gain":
        daily_calories *= 1.2

    return round(daily_calories, 2)

def clean_numeric_value(value):
    """Convert numeric strings to float and handle units (g, mg, mcg)."""
    if isinstance(value, (int, float)):
        return float(value)

    match = re.search(r'([\d.]+)', str(value))
    if match:
        number = float(match.group(1))
        if "mg" in str(value).lower():
            number /= 1000
        elif "mcg" in str(value).lower():
            number /= 1000000
        return number
    return 0.0

def preprocess_input(protein, carbohydrate, fat, dietary_preference):
    """Prepare input for model prediction."""
    return pd.DataFrame({
        'protein': [protein],
        'carbohydrate': [carbohydrate],
        'fat': [fat],
        'fiber': [protein * 0.05],  
        'sugars': [carbohydrate * 0.1],  
        'Category': ["Protein"],
        'Type': ["General"]
    })

def generate_meal_plan(daily_calories, diet_preference):
    """Generate a meal plan based on caloric needs and dietary preferences."""
    meals = 3
    calorie_target_per_meal = daily_calories / meals

    if diet_preference == "vegan":
        filtered_data = data[data["Type"].str.contains("Vegan", na=False)]
    elif diet_preference == "vegetarian":
        filtered_data = data[data["Type"].str.contains("Vegetarian|Vegan", na=False)]
    else:
        filtered_data = data

    proteins = filtered_data[filtered_data["Category"] == "Protein"]
    carbs = filtered_data[filtered_data["Category"] == "Carbs"]
    vegetables = filtered_data[filtered_data["Category"] == "Vegetables"]

    meal_plan = []

    for meal_num in range(1, meals + 1):
        total_meal_calories = 0
        selected_foods = []

        for category, items, initial_weight in [
            ("Protein", proteins, 120),
            ("Carbs", carbs, 150),
            ("Vegetables", vegetables, 100),
        ]:
            if not items.empty:
                food_item = items.sample(n=1).iloc[0]
                weight = initial_weight

                # Extract raw values
                protein_per_100g = clean_numeric_value(food_item["protein"])
                fat_per_100g = clean_numeric_value(food_item["fat"])
                carbs_per_100g = clean_numeric_value(food_item["carbohydrate"])

                # Adjust nutrient values
                adjusted_protein = protein_per_100g * (weight / 100)
                adjusted_fat = fat_per_100g * (weight / 100)
                adjusted_carbs = carbs_per_100g * (weight / 100)

                # Predict calories
                input_data = preprocess_input(
                    adjusted_protein, adjusted_carbs, adjusted_fat, diet_preference
                )
                predicted_calories_per_100g = model.predict(input_data)[0]
                portion_calories = (predicted_calories_per_100g / 100) * weight

                # Add the selected food
                selected_foods.append({
                    "name": food_item["name"],
                    "calories": portion_calories,
                    "protein": round(adjusted_protein, 2),
                    "fats": round(adjusted_fat, 2),
                    "carbs": round(adjusted_carbs, 2),
                    "weight": round(weight, 2)
                })
                total_meal_calories += portion_calories

        scaling_factor = calorie_target_per_meal / total_meal_calories if total_meal_calories > 0 else 1
        for food in selected_foods:
            food["weight"] *= scaling_factor
            food["calories"] *= scaling_factor
            total_meal_calories = calorie_target_per_meal

        meal_plan.append({
            "meal": f"Meal {meal_num}",
            "total_calories": round(total_meal_calories, 2),
            "items": selected_foods
        })

    return meal_plan

@app.route("/", methods=["GET"])
def index():
    return render_template("interface.html")

@app.route("/generate_meal_plan", methods=["POST"])
def generate_meal_plan_api():
    try:
        weight = float(request.form["weight"])
        height = float(request.form["height"])
        age = int(request.form["age"])
        gender = request.form["gender"]
        activity_level = request.form["activity_level"]
        goal = request.form["goal"]
        dietary_preference = request.form["dietary_preference"]

        daily_calories = calculate_daily_calories(weight, height, age, gender, activity_level, goal)
        meal_plan = generate_meal_plan(daily_calories, dietary_preference)

        return jsonify({"daily_calories": daily_calories, "meal_plan": meal_plan})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)

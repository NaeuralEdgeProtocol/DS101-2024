import pandas as pd
import re
import os
import time

file_path = "nutrition.xlsx"
data = pd.read_excel(file_path)

relevant_columns = [
    'name',
    'serving_size',
    'calories',
    'saturated_fat',
    'protein',
    'carbohydrate',
    'fat',
    'fiber',
    'sugars',
    'saturated_fatty_acids',
    'monounsaturated_fatty_acids'
]

data = data[relevant_columns].copy()

def extract_numeric_value(value):
    """Extract numeric value from a string and convert to float, replacing NaN with 0."""
    if pd.isna(value) or value == "":
        return 0.0
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

numeric_columns = [
    'protein', 'carbohydrate', 'fat', 'fiber', 'sugars', 
    'saturated_fatty_acids', 'monounsaturated_fatty_acids', 'calories', 'saturated_fat'
]

original_values = data[numeric_columns].copy()

for col in numeric_columns:
    data[col] = data[col].apply(extract_numeric_value)


def is_healthy_food(row):
    """Determine if food is healthy based on calorie, sugar, and fat content."""
    name_lower = row['name'].lower()
    calories = extract_numeric_value(row['calories'])
    saturated_fat = extract_numeric_value(row['saturated_fat'])
    sugars = extract_numeric_value(row['sugars'])

    fast_food_keywords =  [
        # Major chains
        'mcdonalds', 'mcdonald\'s', 'burger king', 'wendy\'s', 'wendys', 
        'kfc', 'pizza hut', 'dominos', 'domino\'s', 'subway', 'taco bell', 
        'arby\'s', 'arbys', 'popeyes', 'popeye\'s', 'dunkin', 'dunkin\'',
        'chick-fil-a', 'chickfila', 'chick fil a', 'sonic', 'five guys',
        'dairy queen', 'papa john\'s', 'papa johns', 'little caesars',
        'chipotle', 'panera', 'jack in the box', 'hardee\'s', 'hardees',
        'carl\'s jr', 'carls jr', 'white castle', 'in-n-out', 'in n out', 
        'applebee\'s', 'applebees', 'dennys', 'denny\'s', 'olive garden', 
        'carrabba\'s italian grill',
        # Generic fast food items
        'big mac', 'whopper', 'happy meal', 'nugget', 'quarter pounder',
        'chicken sandwich', 'value meal', 'combo meal', 'drive thru',
        'drive-thru', 'fast food', 'takeout', 'take-out', 'take out',
        'school lunch', 'campbell\'s', 'campbells', 'supper bakes meal kits',
        'morningstar farms', 'loma linda',
        # Common fast food descriptors
        'restaurant style', 'copycat', 'fast casual', 'quick serve',
        'food court', 'food-court', 'mall food', 'restaurant',
        'ready-to-eat', 'ready to eat', 'Cornstarch', 'PACE'  
    ]
    unhealthy_keywords =  [
        'fried', 'fries', 'chip', 'candy', 'chocolate', 'soda', 'pop', 
        'processed', 'instant', 'microwave', 'frozen dinner',
        'sugary', 'sweet', 'cake', 'cookie', 'pastry', 'donut',
        'ice cream', 'milkshake', 'deep fried', 'breaded', 'babyfood', 
        'infant formula', 'child formula', 'alcoholic beverage',
        'alcohol', 'beverages', 'ready-to-serve', 'ready to serve', 'eggo',
        'ready-to-eat', 'ready to eat', 'dessert', 'Gelatin desserts',
        'Cornstarch', 'PACE','cornstarch', 'pace','Mayonnaise', 'mayonnaise', 'salad dressing',
        'margarine', 'frostings', 'puddings', 'pudding', 'dry mix', 'chewing gum', 'tapioca', 'flour',
        'corn bran', 'juice', 'spices'
         
    ]

    if any(keyword in name_lower for keyword in fast_food_keywords + unhealthy_keywords):
        return False
    if calories > 500 or saturated_fat > 5 or sugars > 25:
        return False
    return True

def determine_food_type(name, protein, carbs, fat, fiber, sugars):
    """Determine food type based on macronutrients and add Vegetarian/Vegan labels."""
    name_lower = name.lower()
    
    if 'fish' in name_lower or 'salmon' in name_lower:
        return 'Fish'
    if 'chicken' in name_lower or 'turkey' in name_lower:
        return 'Poultry'
    if 'beef' in name_lower or 'pork' in name_lower:
        return 'Red Meat'
    if 'egg' in name_lower:
        return 'Eggs (Vegetarian)'
    if 'bean' in name_lower or 'lentil' in name_lower:
        return 'Legumes (Vegan)'
    if 'milk' in name_lower or 'cheese' in name_lower or 'yogurt' in name_lower:
        if 'greek yogurt' in name_lower or 'cottage cheese' in name_lower:
            return 'High Protein Dairy (Vegetarian)'
        return 'Dairy (Vegetarian)'
    if 'rice' in name_lower or 'bread' in name_lower:
        return 'Grains (Vegan)'
    if 'nut' in name_lower or 'almond' in name_lower:
        return 'Nuts and Seeds (Vegan)'
    if sugars >= 5 and carbs >= 10 and protein < 3 and fat < 1:
        return 'Fruits (Vegan)'
    if protein < 5 and fat < 1 and carbs < 12 and sugars < 5 and fiber > 1:
        return 'Vegetables (Vegan)'

    return 'Other'

def categorize_food(row):
    """Categorize food into major food groups with vegetarian/vegan labeling."""
    protein = row['protein']
    carbs = row['carbohydrate']
    fat = row['fat']
    calories = row['calories']
    fiber = row['fiber']
    sugars = row['sugars']

    if not is_healthy_food(row):
        return None

    total_macros = protein + carbs + fat
    protein_ratio = protein / total_macros if total_macros > 0 else 0

    if protein >= 15 or (protein >= 10 and protein_ratio > 0.3):
        category = 'Protein'
    elif sugars >= 5 and carbs >= 10 and protein < 3 and fat < 1:
        category = 'Fruits'
    elif calories < 50 and fat < 1 and protein < 5 and carbs < 12 and sugars < 5:
        category = 'Vegetables'
    elif fat > protein and fat > carbs and fat >= 10:
        category = 'Fats'
    elif carbs > protein and carbs > fat and carbs >= 15:
        category = 'Carbs'
    else:
        category = 'Mixed'

    food_type = determine_food_type(row['name'], protein, carbs, fat, fiber, sugars)
    return {'Category': category, 'Type': food_type}

categorization = data.apply(categorize_food, axis=1)
healthy_data = data[categorization.notna()].copy()


healthy_data['Category'] = categorization[categorization.notna()].apply(lambda x: x['Category'])
healthy_data['Type'] = categorization[categorization.notna()].apply(lambda x: x['Type'])


healthy_data[numeric_columns] = original_values[categorization.notna()]

def remove_file_safely(file_path):
    attempts = 3
    for i in range(attempts):
        try:
            if os.path.exists(file_path):
                os.rename(file_path, file_path + "_old")  
            return
        except PermissionError:
            print(f"File in use. Retrying in 2 seconds... ({i+1}/{attempts})")
            time.sleep(2)

output_file = "healthy_foods_for_meal_planning.csv"
remove_file_safely(output_file)

with open(output_file, 'w', newline='') as file:
    healthy_data.to_csv(file, index=False)

print("Healthy categorized data saved successfully.")

import pickle
import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from sklearn.metrics.pairwise import cosine_similarity

MODEL_PATH = "../model/"

with open(MODEL_PATH + "sanitized_recipes.json", 'r', encoding='utf-8') as f:
    recipes = json.load(f)

ingredients_set = set()
with open(MODEL_PATH + "sanitized_ingredients.txt", 'r') as f:
    for line in f:
        ingredients_set.add(line.strip().lower())

try:
    with open(MODEL_PATH + "model_file", 'rb') as f:
        loaded_tfidf = pickle.load(f)
    with open(MODEL_PATH + "matrix_file", 'rb') as f:
        loaded_tfidf_matrix = pickle.load(f)
    print("Model and matrix loaded successfully.")

except FileNotFoundError:
    print("Error: Model or matrix file not found.")
except Exception as e:
    print(f"Error loading model or matrix: {e}")

class ReceivedData(BaseModel):
  ingredients: List[str]
  recipes_number_wanted: int
  
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React frontend
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

def lookup_ingredients(all_ingredients, target_ingredients):
    related_ingredients = []
    for ingredient in all_ingredients:
        for target_ingredient in target_ingredients:
            if target_ingredient.lower() in ingredient.lower():  # Case-insensitive check
                related_ingredients.append(ingredient)
    return related_ingredients

def calculate_similarity(user_ingredients, loaded_tfidf, loaded_tfidf_matrix):
    user_ingredient_string = " ".join([ingredient.lower().strip() for ingredient in user_ingredients])
    user_vector = loaded_tfidf.transform([user_ingredient_string]) # Use the loaded tfidf model
    similarity_scores = cosine_similarity(user_vector, loaded_tfidf_matrix) # Use the loaded matrix
    return similarity_scores

def compute_recipies(user_ingredients, cnt_recommendations):
    enriched_user_ingredients = lookup_ingredients(ingredients_set, user_ingredients)
    similarity_scores = calculate_similarity(enriched_user_ingredients, loaded_tfidf, loaded_tfidf_matrix)

    top_n_indices = similarity_scores.argsort()[0][::-1][:cnt_recommendations]

    recommended_recipes = [recipes[i] for i in top_n_indices]

    return recommended_recipes
    
# the appraise endpoint will receive the data from the form and return the inferred price
@app.post("/recommend_recipies")
async def recommend_recipies(data: ReceivedData):
  recommended_recipes = compute_recipies(data.ingredients, data.recipes_number_wanted)
  return {"recommended_recipes": recommended_recipes}
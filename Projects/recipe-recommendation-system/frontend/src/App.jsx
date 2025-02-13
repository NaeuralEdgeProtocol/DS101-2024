import React, { useState } from "react";
import './App.css';

export default function App() {
  const [ingredients, setIngredients] = useState([""]);
  const [recipes, setRecipes] = useState([]);
  const [loading, setLoading] = useState(false);
  const [numRecipes, setNumRecipes] = useState(5);

  const handleInputChange = (index, value) => {
    const newIngredients = [...ingredients];
    newIngredients[index] = value;
    setIngredients(newIngredients);
  };

  const addIngredientField = () => {
    setIngredients([...ingredients, ""]);
  };

  const removeIngredientField = (index) => {
    const newIngredients = ingredients.filter((_, i) => i !== index);
    setIngredients(newIngredients);
  };

  const handleNumRecipesChange = (e) => {
    setNumRecipes(parseInt(e.target.value) || 1);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setRecipes([]);

    try {
      const response = await fetch("http://localhost:5050/recommend_recipies", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ ingredients, recipes_number_wanted: numRecipes }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || response.statusText);
      }

      const data = await response.json();
      setRecipes(data.recommended_recipes);
    } catch (error) {
      console.error("Error fetching recommendations:", error);
      alert("Failed to fetch recommendations. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <div className="card">
        <h1>Recipe Recommendation System</h1>
        <form onSubmit={handleSubmit} className="form-container">
          {ingredients.map((ingredient, index) => (
            <div className="ingredient-field" key={index}>
              <input
                type="text"
                placeholder={`Ingredient ${index + 1}`}
                value={ingredient}
                onChange={(e) => handleInputChange(index, e.target.value)}
                required
              />
              {ingredients.length > 1 && (
                <button
                  type="button"
                  onClick={() => removeIngredientField(index)}
                  className="remove-button"
                >
                  Remove
                </button>
              )}
            </div>
          ))}

          <div style={{ marginBottom: "20px", textAlign: "center" }}>
            <button type="button" onClick={addIngredientField} className="add-button">
              Add Ingredient
            </button>
          </div>

          <div className="num-recipes-field">
            <label htmlFor="numRecipes">Number of Recipes:</label>
            <input
              type="number"
              id="numRecipes"
              value={numRecipes}
              onChange={handleNumRecipesChange}
              min="1"
              required
            />
          </div>

          <div style={{ textAlign: "center" }}>
            <button
              type="submit"
              className={`submit-button ${loading ? 'loading' : ''}`}
              disabled={loading}
            >
              {loading ? "Loading..." : "Get Recipes"}
            </button>
          </div>
        </form>

        {loading === false && recipes.length > 0 && (
          <div className="recipe-list">
            <h2>Recommended Recipes:</h2>
            <ul>
              {recipes.map((recipe, index) => (
                <li key={index}>
                  <h3>{recipe.title}</h3>
                  <p>{recipe.description}</p>
                  <h4>Ingredients:</h4>
                  <ul>
                    {recipe.ingredients.map((ingredient, i) => (
                      <li key={i}>{ingredient}</li>
                    ))}
                  </ul>
                  <h4>Steps:</h4>
                  <ol>
                    {recipe.steps.map((step, i) => (
                      <li key={i}>{step}</li>
                    ))}
                  </ol>
                  <a href={recipe.link} target="_blank" rel="noopener noreferrer">
                    View Original Recipe
                  </a>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
}
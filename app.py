import torch
import json
import warnings
import tkinter as tk
from PIL import Image, ImageTk
from model3 import ImprovedSoccerPredictionModel
from model3 import load_model
from model3 import load_and_preprocess_data
from model3 import make_prediction
from model3 import get_recent_team_stats

warnings.filterwarnings("ignore")

def predict_match_with_recent_form(model, home_team, away_team, df, team_ratings, device, days=365):
    if home_team not in team_ratings or away_team not in team_ratings:
        return "One or both teams not found in database!"
        
    home_stats = get_recent_team_stats(df, home_team, days=days)
    away_stats = get_recent_team_stats(df, away_team, days=days)
    
    dummy_row = {
        'team_home': home_team,
        'team_away': away_team,
        'xG_home': home_stats['xG_home'],
        'xG_away': away_stats['xG_away'],
        'npxGD': home_stats['xG_home'] - away_stats['xG_away'],
        'deep_home': home_stats['deep_home'],
        'deep_away': away_stats['deep_away'],
        'xpts_diff': home_stats['xpts_diff'],
        'xpts_away': away_stats['xpts_away'],
        'home_rolling_xG': home_stats['home_rolling_xG'],
        'away_rolling_xG': away_stats['away_rolling_xG'],
        'home_rolling_score': home_stats['home_rolling_score'],
        'away_rolling_score': away_stats['away_rolling_score'],
        'home_rolling_deep': home_stats['home_rolling_deep'],
        'away_rolling_deep': away_stats['away_rolling_deep'],
        'home_rolling_xpts': home_stats['home_rolling_xpts'],
        'away_rolling_xpts': away_stats['away_rolling_xpts'],
        'home_rolling_xG_diff': home_stats['home_rolling_xG_diff'],
        'away_rolling_xG_diff': away_stats['away_rolling_xG_diff'],
        'home_rolling_xGA_diff': home_stats['home_rolling_xGA_diff'],
        'away_rolling_xGA_diff': away_stats['away_rolling_xGA_diff']
    }
    
    pred, probs = make_prediction(model, dummy_row, team_ratings, device)
    
    max_prob = max(probs)
    
    if max_prob == probs[0]:
        if (probs[0] - probs[2]) < 0.20:
            bet = "1X"
        else:
            bet = "1"
    elif max_prob == probs[2]:
        if (probs[2] - probs[0]) < 0.20:
            bet = "X2"
        else:
            bet = "2"
    else:
        bet = "X"
    
    prediction_text = f"""
    {home_team} vs {away_team}:
    Predicted Score: {float(pred[0]):.2f} - {float(pred[1]):.2f}

    Outcome Probabilities:
    Home Win: {probs[0]:.2%}
    Draw: {probs[1]:.2%}
    Away Win: {probs[2]:.2%}
    Best bet: {bet}
    """
    return prediction_text

root = tk.Tk()
root.title("Football Match Predictor")
root.geometry("1000x600")

bg_image = Image.open("background.jpg")
bg_image = bg_image.resize((1000, 600), Image.ANTIALIAS)
bg_photo = ImageTk.PhotoImage(bg_image)

bg_label = tk.Label(root, image=bg_photo)
bg_label.place(relwidth=1, relheight=1)

title_label = tk.Label(root, text="Predict a match...", font=("Arial", 40, "bold"), fg="white", bg="black")
title_label.pack(pady=10)

def show_prediction_popup(home_team, away_team):
    popup = tk.Toplevel(root)
    popup.title("Match Prediction")
    popup.geometry("600x400")
    
    popup_bg_image = Image.open("ucl.jpg")
    popup_bg_image = popup_bg_image.resize((600, 400), Image.ANTIALIAS)
    popup_bg_photo = ImageTk.PhotoImage(popup_bg_image)
    
    popup_bg_label = tk.Label(popup, image=popup_bg_photo)
    popup_bg_label.place(relwidth=1, relheight=1)
    popup_bg_label.image = popup_bg_photo
    
    prediction_text = predict_match_with_recent_form(model, home_team, away_team, df, team_ratings, device, days=365)
    
    label = tk.Label(popup, text=prediction_text, justify="left", anchor="w", font=("Arial", 12, "bold"), fg="white", bg="black")
    label.pack(padx=10, pady=10)
    
    close_button = tk.Button(popup, text="Close", font=("Arial", 12, "bold"), bg="red", fg="white", command=popup.destroy)
    close_button.pack(pady=10)

def get_input_and_show_popup():
    home_team = home_entry.get().strip()
    away_team = away_entry.get().strip()
    if home_team and away_team:
        error_label.config(text="")
        show_prediction_popup(home_team, away_team)
    else:
        error_label.config(text="Please enter both team names!")

tk.Label(root, text="Home Team:", font=("Arial", 30, "bold"), bg="black", fg="white").pack()
home_entry = tk.Entry(root, font=("Arial", 30))
home_entry.pack(pady=5)

tk.Label(root, text="Away Team:", font=("Arial", 30, "bold"), bg="black", fg="white").pack()
away_entry = tk.Entry(root, font=("Arial", 30))
away_entry.pack(pady=5)

error_label = tk.Label(root, text="", fg="red", font=("Arial", 30, "bold"), bg="black")
error_label.pack()

go_button = tk.Button(root, text="GO", font=("Arial", 36, "bold"), bg="green", fg="white", command=get_input_and_show_popup)
go_button.pack(pady=15)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
df = load_and_preprocess_data('clean_data.csv')
with open('team_ratings.json', 'r') as f:
    team_ratings = json.load(f)

model = ImprovedSoccerPredictionModel(input_size=23).to(device)
load_model(model, 'best_model3.pth')

root.mainloop()
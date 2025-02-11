import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from pulp import *
from tqdm import tqdm

# Custom loss function for soccer predictions
class SoccerLoss(nn.Module):
    def __init__(self, goals_weight=1.0, result_weight=2.0):  
        super().__init__()
        self.goals_weight = goals_weight
        self.result_weight = result_weight
        
    def forward(self, predictions, targets):
        goals_loss = F.mse_loss(predictions[:, :2], targets[:, :2])  # Loss for goal predictions
        result_loss = F.mse_loss(predictions[:, 2], targets[:, 2])   # Loss for match result prediction
        return self.goals_weight * goals_loss + self.result_weight * result_loss

# Load and preprocess the soccer data
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # Calculate time-based weights for matches
    max_date = df['date'].max()
    df['days_from_max'] = (max_date - df['date']).dt.days
    df['time_weight'] = 1 / (1 + np.exp(df['days_from_max'] / 365))
    
    # Add rolling averages for both traditional and new metrics
    for team_type in ['home', 'away']:
        team_col = f'team_{team_type}'
        
        # Original rolling averages
        df[f'{team_type}_rolling_xG'] = df.groupby(team_col)[f'xG_{team_type}'].transform(
            lambda x: x.rolling(5, min_periods=1).mean())
        df[f'{team_type}_rolling_score'] = df.groupby(team_col)[f'scored_{team_type}'].transform(
            lambda x: x.rolling(5, min_periods=1).mean())
        
        # New metrics rolling averages
        df[f'{team_type}_rolling_deep'] = df.groupby(team_col)[f'deep_{team_type}'].transform(
            lambda x: x.rolling(5, min_periods=1).mean())
        df[f'{team_type}_rolling_xpts'] = df.groupby(team_col)[f'xpts_diff_{team_type}'].transform(
            lambda x: x.rolling(5, min_periods=1).mean())
        df[f'{team_type}_rolling_xG_diff'] = df.groupby(team_col)[f'xG_diff_{team_type}'].transform(
            lambda x: x.rolling(5, min_periods=1).mean())
        df[f'{team_type}_rolling_xGA_diff'] = df.groupby(team_col)[f'xGA_diff_{team_type}'].transform(
            lambda x: x.rolling(5, min_periods=1).mean())
    
    return df

# Custom dataset class for soccer data
class SoccerDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

# Neural network model for soccer prediction
class ImprovedSoccerPredictionModel(nn.Module):
    def __init__(self, input_size):
        super(ImprovedSoccerPredictionModel, self).__init__()
        # Define a deeper feed-forward network
        self.fc1 = nn.Linear(input_size, 512)
        self.bn1 = nn.BatchNorm1d(512)
        
        self.fc2 = nn.Linear(512, 512)
        self.bn2 = nn.BatchNorm1d(512)
        
        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        
        self.fc4 = nn.Linear(256, 256)
        self.bn4 = nn.BatchNorm1d(256)
        
        self.fc5 = nn.Linear(256, 128)
        self.bn5 = nn.BatchNorm1d(128)
        
        self.fc6 = nn.Linear(128, 64)
        self.bn6 = nn.BatchNorm1d(64)
        
        # Separate prediction heads for goals and match result
        self.goals_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # Predict [home_goals, away_goals]
        )
        
        self.result_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Predict scalar result (using tanh later)
        )
        
        self.dropout = nn.Dropout(0.3)  # Dropout for regularization
    
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        
        x = F.relu(self.bn4(self.fc4(x)))
        x = self.dropout(x)
        
        x = F.relu(self.bn5(self.fc5(x)))
        x = self.dropout(x)
        
        x = F.relu(self.bn6(self.fc6(x)))
        
        # Separate predictions for goals and match result
        goals = self.goals_head(x)
        result = torch.tanh(self.result_head(x))
        
        return torch.cat([goals, result], dim=1)  # Concatenate outputs

# Normalize features for better model performance
def normalize_features(features):
    features_array = np.array(features)
    mean = np.mean(features_array, axis=0)
    std = np.std(features_array, axis=0)
    return (features_array - mean) / (std + 1e-8), mean, std

# Calculate team ratings using linear programming
def calculate_team_ratings(df):
    print("Calculating team ratings...")
    teams = pd.concat([df['team_home'], df['team_away']]).unique()
    print(f"Number of teams: {len(teams)}")
    prob = LpProblem("Soccer_Ratings", LpMinimize)
    
    # Variables
    team_off_rating = LpVariable.dicts("off", teams, lowBound=-3, upBound=3)
    team_def_rating = LpVariable.dicts("def", teams, lowBound=-3, upBound=3)
    home_advantage = LpVariable("home_advantage", lowBound=0, upBound=1)
    
    abs_diff_home = LpVariable.dicts("abs_diff_home", range(len(df)), lowBound=0)
    abs_diff_away = LpVariable.dicts("abs_diff_away", range(len(df)), lowBound=0)
    
    # Objective function and constraints
    prob += lpSum([df.iloc[i]['time_weight'] * (abs_diff_home[i] + abs_diff_away[i]) 
                  for i in range(len(df))])
    
    for i in range(len(df)):
        row = df.iloc[i]
        home_expected = (team_off_rating[row['team_home']] - 
                        team_def_rating[row['team_away']] + 
                        home_advantage)
        away_expected = (team_off_rating[row['team_away']] - 
                        team_def_rating[row['team_home']] - 
                        home_advantage)
        
        prob += abs_diff_home[i] >= home_expected - row['xG_home']
        prob += abs_diff_home[i] >= -(home_expected - row['xG_home'])
        prob += abs_diff_away[i] >= away_expected - row['xG_away']
        prob += abs_diff_away[i] >= -(away_expected - row['xG_away'])
    
    # Solve with minimal output
    prob.solve(PULP_CBC_CMD(msg=0))
    print("Team ratings calculated successfully!")
    
    # Create ratings dictionary
    ratings = {}
    for team in teams:
        off_rating = value(team_off_rating[team])
        def_rating = value(team_def_rating[team])
        ratings[team] = {'offensive': off_rating, 'defensive': def_rating}
    
    return ratings, value(home_advantage)

# Compute Poisson probability for goal predictions
def poisson_probability(k, lam):
    return math.exp(-lam) * (lam ** k) / math.factorial(k)

# Compute outcome probabilities using a modified Poisson model
def compute_outcome_probabilities(lambda_home, lambda_away, max_goals=10):
    home_win = 0.0
    draw = 0.0
    away_win = 0.0

    for home_goals in range(max_goals):
        for away_goals in range(max_goals):
            base_p = poisson_probability(home_goals, lambda_home) * poisson_probability(away_goals, lambda_away)
            
            goal_diff = abs(home_goals - away_goals)
            if goal_diff == 0:
                similarity_factor = 1 + 1/(1 + abs(lambda_home - lambda_away))
                base_p *= similarity_factor
            else:
                base_p *= math.exp(-goal_diff * 0.1)
            
            if home_goals > away_goals:
                home_win += base_p
            elif home_goals == away_goals:
                draw += base_p
            else:
                away_win += base_p

    total = home_win + draw + away_win
    return home_win / total, draw / total, away_win / total

# Make predictions using the trained model
def make_prediction(model, input_data, team_ratings, device):
    model.eval()
    with torch.no_grad():
        feature = torch.FloatTensor([
            team_ratings[input_data['team_home']]['offensive'],
            team_ratings[input_data['team_home']]['defensive'],
            team_ratings[input_data['team_away']]['offensive'],
            team_ratings[input_data['team_away']]['defensive'],
            input_data['xG_home'],
            input_data['xG_away'],
            input_data['npxGD'],
            input_data['deep_home'],
            input_data['deep_away'],
            input_data['xpts_diff'],
            input_data['xpts_away'],
            input_data['home_rolling_xG'],
            input_data['away_rolling_xG'],
            input_data['home_rolling_score'],
            input_data['away_rolling_score'],
            input_data['home_rolling_deep'],
            input_data['away_rolling_deep'],
            input_data['home_rolling_xpts'],
            input_data['away_rolling_xpts'],
            input_data['home_rolling_xG_diff'],
            input_data['away_rolling_xG_diff'],
            input_data['home_rolling_xGA_diff'],
            input_data['away_rolling_xGA_diff']
        ]).unsqueeze(0).to(device)
        
        prediction = model(feature)
        lambda_home = F.softplus(prediction[0, 0]).item()
        lambda_away = F.softplus(prediction[0, 1]).item()
        win_prob, draw_prob, loss_prob = compute_outcome_probabilities(lambda_home, lambda_away)
        
        return prediction.cpu().numpy()[0], (win_prob, draw_prob, loss_prob)

# Print team ratings
def print_team_ratings(ratings):
    print("\nTeam Ratings:")
    print("-" * 60)
    print(f"{'Team':<30} {'Offensive':>10} {'Defensive':>10}")
    print("-" * 60)
    for team, rating in sorted(ratings.items()):
        print(f"{team:<30} {rating['offensive']:>10.3f} {rating['defensive']:>10.3f}")

# Print match prediction results
def print_match_prediction(model, row, team_ratings, device):
    pred, probs = make_prediction(model, row, team_ratings, device)
    print(f"\nMatch: {row['team_home']} vs {row['team_away']}")
    print(f"Actual Score: {row['scored_home']}-{row['scored_away']} (Result: {row['result']})")
    print(f"Predicted Score: {float(pred[0]):.2f} - {float(pred[1]):.2f}")
    print("Outcome Probabilities:")
    print(f"Home Win: {probs[0]:.2%}")
    print(f"Draw: {probs[1]:.2%}")
    print(f"Away Win: {probs[2]:.2%}")
    
    result_pred = 'w' if probs[0] > max(probs[1], probs[2]) else 'd' if probs[1] > max(probs[0], probs[2]) else 'l'
    actual_result = row['result']
    predicted_correct = result_pred == actual_result
    print(f"Prediction {'Correct' if predicted_correct else 'Incorrect'}")
    print("-" * 60)
    return predicted_correct

# Get recent team statistics
def get_recent_team_stats(df, team, last_n_matches=10, days=365):
    current_date = df['date'].max()
    recent_df = df[df['date'] > (current_date - pd.Timedelta(days=days))]
    
    home_matches = recent_df[recent_df['team_home'] == team].tail(last_n_matches)
    away_matches = recent_df[recent_df['team_away'] == team].tail(last_n_matches)
    
    stats = {
        'xG_home': home_matches['xG_home'].mean() if not home_matches.empty else df['xG_home'].mean(),
        'xG_away': away_matches['xG_away'].mean() if not away_matches.empty else df['xG_away'].mean(),
        'deep_home': home_matches['deep_home'].mean() if not home_matches.empty else df['deep_home'].mean(),
        'deep_away': away_matches['deep_away'].mean() if not away_matches.empty else df['deep_away'].mean(),
        'xpts_diff': home_matches['xpts_diff'].mean() if not home_matches.empty else df['xpts_diff'].mean(),
        'xpts_away': away_matches['xpts_away'].mean() if not away_matches.empty else df['xpts_away'].mean(),
        'scored_home': home_matches['scored_home'].mean() if not home_matches.empty else df['scored_home'].mean(),
        'scored_away': away_matches['scored_away'].mean() if not away_matches.empty else df['scored_away'].mean(),
        
        'home_rolling_xG': home_matches['home_rolling_xG'].mean() if not home_matches.empty else 0,
        'away_rolling_xG': away_matches['away_rolling_xG'].mean() if not away_matches.empty else 0,
        'home_rolling_score': home_matches['home_rolling_score'].mean() if not home_matches.empty else 0,
        'away_rolling_score': away_matches['away_rolling_score'].mean() if not away_matches.empty else 0,
        'home_rolling_deep': home_matches['home_rolling_deep'].mean() if not home_matches.empty else 0,
        'away_rolling_deep': away_matches['away_rolling_deep'].mean() if not away_matches.empty else 0,
        'home_rolling_xpts': home_matches['home_rolling_xpts'].mean() if not home_matches.empty else 0,
        'away_rolling_xpts': away_matches['away_rolling_xpts'].mean() if not away_matches.empty else 0,
        'home_rolling_xG_diff': home_matches['home_rolling_xG_diff'].mean() if not home_matches.empty else 0,
        'away_rolling_xG_diff': away_matches['away_rolling_xG_diff'].mean() if not away_matches.empty else 0,
        'home_rolling_xGA_diff': home_matches['home_rolling_xGA_diff'].mean() if not home_matches.empty else 0,
        'away_rolling_xGA_diff': away_matches['away_rolling_xGA_diff'].mean() if not away_matches.empty else 0
    }
    
    return stats

# Predict match outcome using recent form
def predict_match_with_recent_form(model, home_team, away_team, df, team_ratings, device, days=365):
    if home_team not in team_ratings or away_team not in team_ratings:
        print(f"One or both teams not found in database!")
        return
        
    home_stats = get_recent_team_stats(df, home_team, days=days)
    away_stats = get_recent_team_stats(df, away_team, days=days)
    
    dummy_row = {
        # Create feature row using recent statistics for both teams
        'team_home': home_team,
        'team_away': away_team,
        'xG_home': home_stats['xG_home'],
        'xG_away': away_stats['xG_away'],
        'npxGD': home_stats['xG_home'] - away_stats['xG_away'],
        'deep_home': home_stats['deep_home'],
        'deep_away': away_stats['deep_away'],
        'xpts_diff': home_stats['xpts_diff'],
        'xpts_away': away_stats['xpts_away'],
        # Include rolling averages for more context
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
    
    # Make prediction using the model
    pred, probs = make_prediction(model, dummy_row, team_ratings, device)
    
    # Print detailed prediction results
    print(f"\nPrediction for {home_team} vs {away_team}:")
    print(f"Predicted Score: {float(pred[0]):.2f} - {float(pred[1]):.2f}")
    print("Outcome Probabilities:")
    print(f"Home Win: {probs[0]:.2%}")
    print(f"Draw: {probs[1]:.2%}")
    print(f"Away Win: {probs[2]:.2%}")
    
    # Print recent form statistics
    print("\nRecent Form:")
    print(f"{home_team}:")
    print(f"  Home xG: {home_stats['xG_home']:.2f}")
    print(f"  Home Scored: {home_stats['scored_home']:.2f}")
    print(f"  Deep Completions: {home_stats['deep_home']:.2f}")
    print(f"  xPts Difference: {home_stats['xpts_diff']:.2f}")
    print(f"{away_team}:")
    print(f"  Away xG: {away_stats['xG_away']:.2f}")
    print(f"  Away Scored: {away_stats['scored_away']:.2f}")
    print(f"  Deep Completions: {away_stats['deep_away']:.2f}")
    print(f"  xPts: {away_stats['xpts_away']:.2f}")

# Train the model with monitoring of training and validation losses
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device):
    best_val_loss = float('inf')
    
    # Training loop for specified number of epochs
    for epoch in tqdm(range(num_epochs), desc="Training Progress"):
        # Training phase
        model.train()
        train_loss = 0
        train_goals_loss = 0
        train_result_loss = 0
        
        # Process each batch in training data
        for batch_features, batch_targets in train_loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(batch_features)
            
            # Calculate separate losses for goals and result
            goals_loss = F.mse_loss(outputs[:, :2], batch_targets[:, :2])
            result_loss = F.mse_loss(outputs[:, 2], batch_targets[:, 2])
            loss = criterion(outputs, batch_targets)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Accumulate losses
            train_loss += loss.item()
            train_goals_loss += goals_loss.item()
            train_result_loss += result_loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_goals_loss = 0
        val_result_loss = 0
        
        # Process validation data without gradient calculation
        with torch.no_grad():
            for batch_features, batch_targets in val_loader:
                batch_features = batch_features.to(device)
                batch_targets = batch_targets.to(device)
                
                outputs = model(batch_features)
                goals_loss = F.mse_loss(outputs[:, :2], batch_targets[:, :2])
                result_loss = F.mse_loss(outputs[:, 2], batch_targets[:, 2])
                loss = criterion(outputs, batch_targets)
                
                val_loss += loss.item()
                val_goals_loss += goals_loss.item()
                val_result_loss += result_loss.item()
        
        # Calculate average losses
        avg_train_loss = train_loss/len(train_loader)
        avg_train_goals_loss = train_goals_loss/len(train_loader)
        avg_train_result_loss = train_result_loss/len(train_loader)
        
        avg_val_loss = val_loss/len(val_loader)
        avg_val_goals_loss = val_goals_loss/len(val_loader)
        avg_val_result_loss = val_result_loss/len(val_loader)
        
        # Update learning rate based on validation loss
        scheduler.step(avg_val_loss)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_model(model, 'best_model.pth')
        
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f'\nEpoch [{epoch+1}/{num_epochs}]')
            print(f'Goals Loss - Train: {avg_train_goals_loss:.4f}, Val: {avg_val_goals_loss:.4f}')
            print(f'Result Loss - Train: {avg_train_result_loss:.4f}, Val: {avg_val_result_loss:.4f}')
            print(f'Total Loss - Train: {avg_train_loss:.4f}, Val: {avg_val_loss:.4f}')

# Save model state dictionary to file
def save_model(model, path):
    torch.save(model.state_dict(), path)

# Load model state dictionary from file
def load_model(model, path):
    model.load_state_dict(torch.load(path, weights_only=True))

# Main execution function
def main():
    # Set device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load and preprocess the dataset
    df = load_and_preprocess_data('clean_data.csv')
    team_ratings, home_advantage = calculate_team_ratings(df)

    with open('team_ratins.json', 'w') as json_file:
        json.dump(team_ratings, json_file)
    
    # Prepare features and targets for model training
    features = []
    targets = []
    
    # Extract features and targets from DataFrame
    for _, row in df.iterrows():
        feature = [
            # Team ratings features
            team_ratings[row['team_home']]['offensive'],
            team_ratings[row['team_home']]['defensive'],
            team_ratings[row['team_away']]['offensive'],
            team_ratings[row['team_away']]['defensive'],
            # Match statistics features
            row['xG_home'],
            row['xG_away'],
            row['npxGD'],
            row['deep_home'],
            row['deep_away'],
            row['xpts_diff'],
            row['xpts_away'],
            # Rolling average features
            row['home_rolling_xG'],
            row['away_rolling_xG'],
            row['home_rolling_score'],
            row['away_rolling_score'],
            row['home_rolling_deep'],
            row['away_rolling_deep'],
            row['home_rolling_xpts'],
            row['away_rolling_xpts'],
            row['home_rolling_xG_diff'],
            row['away_rolling_xG_diff'],
            row['home_rolling_xGA_diff'],
            row['away_rolling_xGA_diff']
        ]
        features.append(feature)
        
        # Map match results to numerical values
        result_map = {'w': 1, 'd': 0, 'l': -1}
        target = [row['scored_home'], row['scored_away'], result_map[row['result']]]
        targets.append(target)
    
    # Normalize features and split data
    features_normalized, feat_mean, feat_std = normalize_features(features)
    X_train, X_val, y_train, y_val = train_test_split(
        features_normalized, targets, test_size=0.2, random_state=42
    )
    
    # Create datasets and dataloaders
    train_dataset = SoccerDataset(X_train, y_train)
    val_dataset = SoccerDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)
    
    # Initialize model and training components
    model = ImprovedSoccerPredictionModel(input_size=len(features[0])).to(device)
    criterion = SoccerLoss(goals_weight=1.0, result_weight=2.0)
    
    # Configure optimizer with stable settings
    optimizer = optim.AdamW(
        model.parameters(),
        lr=0.0005,
        weight_decay=1e-4,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Configure learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=3,
        factor=0.5,
        verbose=True,
        min_lr=1e-6
    )
    
    # Train the model
    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=50, device=device)
    
    # Load best model and evaluate performance
    load_model(model, 'best_model.pth')
    print_team_ratings(team_ratings)
    
    # Make sample predictions
    print("\nExample Predictions:")
    print("=" * 60)
    
    sample_matches = df.sample(n=5)
    correct_predictions = 0
    for _, match in sample_matches.iterrows():
        is_correct = print_match_prediction(model, match, team_ratings, device)
        if is_correct:
            correct_predictions += 1
    
    print(f"\nSample Prediction Accuracy: {correct_predictions/5:.2%}")
    
    # Predict specific high-profile matches
    print("\nPredicting matches using recent form:")
    print("=" * 60)
    predict_match_with_recent_form(model, "Barcelona", "Real Madrid", df, team_ratings, device, days=365)
    predict_match_with_recent_form(model, "Manchester City", "Liverpool", df, team_ratings, device, days=365)
    predict_match_with_recent_form(model, "Bayern Munich", "Borussia Dortmund", df, team_ratings, device, days=365)
    
    # Evaluate overall model performance
    print("\nOverall Model Performance:")
    print("=" * 60)
    
    val_correct = 0
    val_total = 0
    val_goal_diff_error = 0
    
    # Calculate validation metrics
    model.eval()
    with torch.no_grad():
        for features, targets in val_loader:
            features = features.to(device)
            targets = targets.to(device)
            
            outputs = model(features)
            
            # Calculate result accuracy
            pred_results = (outputs[:, 2] > 0.5).float() * 2 - 1
            actual_results = targets[:, 2]
            val_correct += (pred_results == actual_results).sum().item()
            val_total += targets.size(0)
            
            # Calculate goal prediction error
            val_goal_diff_error += F.l1_loss(outputs[:, :2], targets[:, :2]).item()
    
    # Print final performance metrics
    print(f"Validation Accuracy: {val_correct/val_total:.2%}")
    print(f"Average Goal Prediction Error: {val_goal_diff_error/len(val_loader):.3f}")

# Entry point of the program
if __name__ == "__main__":
    main()
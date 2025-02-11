import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import re

#Load preprocessed dataset
file_path = "healthy_foods_for_meal_planning.csv"
data = pd.read_csv(file_path)

#Selecting features and target
numeric_features = ['protein', 'carbohydrate', 'fat', 'fiber', 'sugars']
categorical_features = ['Category', 'Type']
target = 'calories'

#Function to clean numeric values and remove units (e.g., 'mg', 'g')
def clean_numeric_value(value):
    if pd.isna(value) or value == "":  # Check if value is missing
        return 0.0  #Replace missing values with 0
    if isinstance(value, (int, float)):  # If it's already numeric
        return float(value)

    #Extracting the numeric part from string using regex
    match = re.search(r'([\d.]+)', str(value))
    if match:
        number = float(match.group(1))  # Convert to float

        #Converting units if necessary
        if "mg" in str(value).lower():
            number = number / 1000  #Convert mg to g
        elif "mcg" in str(value).lower():
            number = number / 1000000  #Convert mcg to g

        return number

    return 0.0  # Default to 0 if no number found

#Applying cleaning function to numeric features
for col in numeric_features:
    data[col] = data[col].apply(clean_numeric_value)

data[target] = data[target].apply(clean_numeric_value)

#Preprocessing pipeline
numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

#Defining Random Forest Regressor
rf_model = RandomForestRegressor(random_state=42)
pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', rf_model)])

# Spliting dataset
X = data[numeric_features + categorical_features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Defining hyperparameter space for RandomizedSearchCV
param_dist = {
    'regressor__n_estimators': [50, 100, 200, 300],
    'regressor__max_depth': [None, 10, 20, 30, 50],
    'regressor__min_samples_split': [2, 5, 10],
    'regressor__min_samples_leaf': [1, 2, 4]
}

# Perform RandomizedSearchCV
tuned_model = RandomizedSearchCV(
    pipeline,
    param_distributions=param_dist,
    n_iter=10,  # Number of random parameter combinations to try
    scoring='neg_root_mean_squared_error',
    cv=5,  # 5-fold cross-validation
    n_jobs=-1,  # Use all available CPU cores
    random_state=42
)

print("Starting RandomizedSearchCV tuning...")
tuned_model.fit(X_train, y_train)
print("Best parameters found:", tuned_model.best_params_)

#Evaluate the tuned model
y_pred = tuned_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"RMSE after tuning: {rmse:.2f}")

best_pipeline = tuned_model.best_estimator_
rf = best_pipeline.named_steps['regressor']
tree = rf.estimators_[0]

def build_limited_graph(tree, feature_names, max_depth=3):
    G = nx.DiGraph()
    def add_edges(node, depth):
        if depth >= max_depth or (tree.children_left[node] == -1 and tree.children_right[node] == -1):
            return
        feature = feature_names[tree.feature[node]]
        threshold = tree.threshold[node]
        left_child = tree.children_left[node]
        right_child = tree.children_right[node]
        G.add_edge(node, left_child, label=f"{feature} <= {threshold:.2f}")
        G.add_edge(node, right_child, label=f"{feature} > {threshold:.2f}")
        add_edges(left_child, depth + 1)
        add_edges(right_child, depth + 1)
    add_edges(0, 0)
    return G

max_depth = 3

feature_names = numeric_features + list(best_pipeline.named_steps['preprocessor']
                                        .transformers_[1][1]
                                        .named_steps['onehot']
                                        .get_feature_names_out(categorical_features))

G = build_limited_graph(tree.tree_, feature_names, max_depth=max_depth)

plt.figure(figsize=(18, 14))
pos = nx.spring_layout(G)  
nx.draw(G, pos, with_labels=True, node_size=2000, font_size=8, edge_color='gray', node_color='lightblue')
edge_labels = nx.get_edge_attributes(G, 'label')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)
plt.title(f"Decision Path in a Single Tree from Random Forest (Max Depth: {max_depth})")
plt.show()

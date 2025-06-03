from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_classification

# Create a dummy dataset
X, y = make_classification(n_samples=100, n_features=20, random_state=42)

# Define the pipeline
pipe = Pipeline([
    ('scaler', None),
    ('classifier', LogisticRegression())
])

# Define the parameter grid
param_grid = {
    'scaler': [StandardScaler(), MinMaxScaler(), RobustScaler()],
    'classifier__C': [0.1, 1.0, 10.0]
}

# Instantiate GridSearchCV
grid = GridSearchCV(pipe, param_grid, cv=3)

# Fit the grid search
grid.fit(X, y)
print(grid.cv_results_)
# Get the best estimator
best_pipe = grid.best_estimator_

# Print the best parameters
print(grid.best_params_)

# Use the best model to make predictions
predictions = best_pipe.predict(X)
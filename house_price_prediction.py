import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load and Clean Data
def load_and_clean_data(filepath):
    data = pd.read_csv(filepath)
    print("Columns in the dataset:", data.columns)  # Print the column names to verify
    # Drop rows with missing values
    data.dropna(inplace=True)
    # Select numerical features excluding 'price'
    features = data.select_dtypes(include=['float64', 'int64']).drop('price', axis=1)
    # Target column is 'price'
    target = data['price']
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


# Step 2: Train the Model
def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# Step 3: Evaluate the Model
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    return mse, r2, predictions

# Step 4: Visualize Results
def plot_results(y_test, predictions):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, predictions, alpha=0.7, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel('Actual Prices')
    plt.ylabel('Predicted Prices')
    plt.title('Actual vs Predicted House Prices')
    plt.show()

# Main Execution
if __name__ == "__main__":
    # Filepath to the dataset
    DATA_PATH = 'data/Housing.csv'  # Ensure the dataset is placed in the `data` folder

    # Step 1: Load and preprocess data
    X_train, X_test, y_train, y_test = load_and_clean_data(DATA_PATH)

    # Step 2: Train the linear regression model
    model = train_linear_regression(X_train, y_train)

    # Step 3: Evaluate the model
    mse, r2, predictions = evaluate_model(model, X_test, y_test)
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R-squared: {r2:.2f}")

    # Step 4: Visualize the results
    plot_results(y_test, predictions)

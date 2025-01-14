# OIBSIP-PROJECT-1-PROPOSAL-LEVEL-2

# House Price Prediction using Linear Regression

## **Project Overview**
This project involves building a predictive model using linear regression to estimate house prices based on various features such as the number of bedrooms, bathrooms, area, parking availability, and more. By leveraging a clean and structured dataset, the model aims to provide accurate predictions and insights into the factors that influence house pricing.

## **Key Features**
- **Data Preprocessing**: Handled missing values and selected relevant numerical features for the model.
- **Model Training**: Utilized the Linear Regression algorithm from Scikit-Learn to train the model.
- **Evaluation Metrics**: Assessed model performance using Mean Squared Error (MSE) and R-squared (R²) score.
- **Visualization**: Plotted actual vs. predicted house prices to visualize model accuracy.

## **Technologies Used**
- **Python**: For scripting and model development.
- **Pandas**: For data manipulation and cleaning.
- **NumPy**: For numerical computations.
- **Matplotlib**: For visualizing the results.
- **Scikit-Learn**: For machine learning implementation.

## **Dataset**
The dataset used in this project contains various features affecting house prices, such as:
- `price` (target variable): The price of the house.
- `area`: Total area of the property.
- `bedrooms`: Number of bedrooms.
- `bathrooms`: Number of bathrooms.
- `stories`: Number of floors.
- `mainroad`, `guestroom`, `basement`, `hotwaterheating`, `airconditioning`, `parking`, `prefarea`, `furnishingstatus`: Additional attributes influencing house value.

## **Workflow**
1. **Data Loading and Exploration**: Loaded the dataset, explored the structure, and identified relevant columns.
2. **Data Cleaning**: Addressed missing values and ensured data quality.
3. **Feature Selection**: Selected numerical features contributing to the target variable (`price`).
4. **Model Training**: Split the data into training and testing sets, then trained a Linear Regression model.
5. **Model Evaluation**: Evaluated the model's performance using MSE and R² metrics.
6. **Result Visualization**: Plotted the relationship between actual and predicted prices for better insights.

## **Performance**
- **Mean Squared Error (MSE)**: Indicates the average squared difference between the predicted and actual values.
- **R-squared (R²)**: Measures how well the model explains the variance in the target variable.

## **How to Run**
1. Clone this repository:
   ```bash
   git clone https://github.com/phanisurya-7/OIBSIP-PROJECT-1-PROPOSAL-LEVEL-2.git
   ```
2. Navigate to the project directory:
   ```bash
   cd House_Price_Prediction
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the script:
   ```bash
   python house_price_prediction.py
   ```

## **Visualization**
The project includes a scatter plot to visualize the relationship between actual and predicted house prices:
- **Blue Points**: Individual predictions.
- **Black Line**: Perfect prediction line where actual = predicted.

## **Future Improvements**
- Include more advanced regression techniques such as Ridge or Lasso regression.
- Explore feature engineering to improve model accuracy.
- Integrate categorical variables using one-hot encoding.
- Deploy the model as a web app for real-time house price prediction.

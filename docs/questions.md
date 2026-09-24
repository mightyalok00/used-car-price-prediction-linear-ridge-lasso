# Used Car Price Prediction: 16 Questions

Dataset: Kaggle — Used Car Listings: Features and Price Prediction  
Target: `price`  
Models: Multiple Linear Regression, Ridge Regression, Lasso Regression

## Data Understanding and Preparation

### Q1. Dataset Understanding and Data Structures

Load the dataset and perform a complete initial inspection. Display its shape, column names, data types, first and last records, descriptive statistics, number of unique values, and memory usage. Explain the main Python/Pandas data structures being used, such as DataFrame, Series, lists, dictionaries, and NumPy arrays, wherever applicable.

### Q2. Data Quality and Cleaning

Identify missing values, duplicate records, invalid values, inconsistent categories, and other data-quality problems. Apply appropriate cleaning techniques and explain why each cleaning decision was made.

### Q3. Feature Classification and Selection

Classify the dataset columns into numerical, categorical, ordinal, identifier, and target variables. Determine which features should be retained, transformed, engineered, or removed before model training and justify your decisions.

## Exploratory Data Analysis

### Q4. Price Distribution and Outlier Analysis

Analyze the distribution of `price` and important numerical features such as mileage, model year/vehicle age, engine characteristics, and other available numerical variables. Detect outliers using appropriate statistical techniques and investigate whether transformations such as `log(price)` improve the distribution.

### Q5. Numerical Features vs Price

Investigate relationships between numerical features and `price` using correlation analysis and appropriate visualizations. Identify which numerical variables have the strongest positive and negative relationships with vehicle price and explain the findings.

### Q6. Categorical Features vs Price

Analyze how available categorical variables such as make/brand, model, fuel type, transmission, drivetrain, accident history, and other vehicle characteristics affect average and median used-car prices. Identify important pricing patterns.

## Data Preprocessing

### Q7. Encoding, Scaling and Data Leakage Prevention

Apply appropriate encoding methods to categorical variables and scaling techniques to numerical variables. Explain why feature scaling is particularly important when comparing Linear, Ridge, and Lasso Regression. Build the preprocessing workflow so information from the test dataset does not leak into model training.

## Multiple Linear Regression

### Q8. Build the Baseline Linear Regression Model

Split the dataset into training and testing sets and develop a Multiple Linear Regression model as the baseline. Evaluate the model using MAE, MSE, RMSE, and R² and explain what each metric indicates about prediction performance.

### Q9. Linear Regression Interpretation and Diagnostics

Analyze the coefficients produced by Multiple Linear Regression to determine which features increase or decrease predicted car prices. Perform residual analysis and investigate relevant Linear Regression assumptions, including linearity, multicollinearity, homoscedasticity, and residual behavior.

## Ridge and Lasso Regression

### Q10. Ridge Regression and Hyperparameter Tuning

Develop a Ridge Regression model using L2 regularization. Use cross-validation and techniques such as GridSearchCV to determine the optimal value of alpha (λ). Analyze how different alpha values affect model coefficients, prediction performance, and overfitting.

### Q11. Lasso Regression and Feature Selection

Develop a Lasso Regression model using L1 regularization and determine the optimal alpha using cross-validation. Identify which coefficients become exactly zero and explain how Lasso can automatically perform feature selection.

## Model Evaluation and Comparison

### Q12. Cross-Validation of All Three Models

Evaluate Multiple Linear Regression, Ridge Regression, and Lasso Regression using the same training/testing split and cross-validation strategy. Calculate the mean and standard deviation of cross-validation performance and investigate whether the results are stable across folds.

### Q13. Comprehensive Linear vs Ridge vs Lasso Comparison

Create a final comparison of all three models using MAE, MSE, RMSE, R², CV mean, CV standard deviation, train R², test R², best alpha, active features, and zero coefficients. Compare predictive errors, training vs testing performance, cross-validation stability, coefficient behavior, regularization effects, overfitting/underfitting, generalization, feature selection, and computational considerations. Explain what the observed results reveal about the practical differences between the three regression techniques.

## Business Analysis

### Q14. Business Analysis — Vehicle Valuation

Based on EDA and model coefficients, identify the vehicle characteristics most strongly associated with used-car value. Explain how a dealership could use these findings when purchasing vehicles, determining listing prices, negotiating trade-ins, and selecting inventory for resale.

### Q15. Business Analysis — Depreciation and Inventory Strategy

Analyze how vehicle age/model year and mileage relate to price depreciation. Investigate whether particular brands, models, fuel types, or other vehicle segments retain value better than others. Explain how these findings could help a dealership make inventory-acquisition decisions.

### Q16. Business Analysis — Pricing Strategy and Opportunities

Compare actual listing prices with model-predicted prices and calculate the prediction difference for each vehicle. Identify potentially underpriced and overpriced listings and analyze how a dealership could use this information to find purchasing opportunities or improve pricing decisions. Discuss the limitations, risks, and factors not captured by the models that should be considered before making real business decisions.

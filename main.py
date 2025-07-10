from sklearn.datasets import fetch_california_housing
import pandas as pd

# Load California Housing dataset
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['PRICE'] = data.target

# Print the first few rows
print(df.head())

import matplotlib.pyplot as plt
import seaborn as sns

# Show dataset info
print(df.info())

# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation with Price")
plt.show()

# Scatterplot: Price vs Median Income
plt.figure(figsize=(6, 4))
sns.scatterplot(x=df['MedInc'], y=df['PRICE'])
plt.title("Price vs Median Income")
plt.xlabel("Median Income")
plt.ylabel("Price")
plt.show()


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Features and target
X = df.drop('PRICE', axis=1)
y = df['PRICE']

# Split dataset: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate model
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"R2 Score: {r2:.3f}")
print(f"RMSE: {rmse:.3f}")


import joblib

# Save the trained model to a file
joblib.dump(model, 'house_price_model.pkl')
print("Model saved successfully!")

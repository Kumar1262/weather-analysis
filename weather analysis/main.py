import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Step 1: Load the Data
df = pd.read_csv('.venv/weather.csv')

# Step 2: Data Exploration
print(df.head())
print(df.info())
print(df.describe())

# Step 3: Data Visualization
sns.set(style="whitegrid")

# Scatter plot: MinTemp vs MaxTemp colored by Rainfall
plt.figure(figsize=(12, 6))
sns.scatterplot(x='MinTemp', y='MaxTemp', hue='Rainfall', data=df, palette='coolwarm', edgecolor='w', alpha=0.7)
plt.title('Relationship between MinTemp and MaxTemp')
plt.show()

# Histogram of Rainfall
plt.figure(figsize=(10, 6))
sns.histplot(df['Rainfall'], bins=30, kde=True, color='skyblue')
plt.title('Distribution of Rainfall')
plt.xlabel('Rainfall (mm)')
plt.ylabel('Frequency')
plt.show()

# Step 4: Feature Engineering (if needed)
# Extracting Month from Date
df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.month

# Step 5: Data Analysis
# Monthly average MaxTemp
monthly_avg_max_temp = df.groupby('Month')['MaxTemp'].mean()

# Step 6: Data Visualization (Part 2)
plt.figure(figsize=(10, 5))
plt.plot(monthly_avg_max_temp.index, monthly_avg_max_temp.values, marker='o')
plt.xlabel('Month')
plt.ylabel('Average Max Temperature (C)')
plt.title('Monthly Average Max Temperature')
plt.grid(True)
plt.show()

# Step 7: Advanced Analysis - Predict Rainfall
# Prepare the data for prediction
X = df[['MinTemp', 'MaxTemp']]
y = df['Rainfall']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions and calculate Mean Squared Error
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error for Rainfall Prediction: {mse:.2f} mm^2')

# Step 8: Conclusions and Insights
highest_rainfall_month = monthly_avg_max_temp.idxmax()
lowest_rainfall_month = monthly_avg_max_temp.idxmin()

print("\nInsights:")
print(f"- The dataset contains {len(df)} records of daily weather data.")
print(f"- Rainfall varies widely, with a mean of {df['Rainfall'].mean():.2f} mm.")
print(f"- Highest average MaxTemp occurs in month {highest_rainfall_month}.")
print(f"- Linear Regression model predicts rainfall with an MSE of {mse:.2f} mm^2.")

# Step 9: Communication (Optional)
# Displaying insights and conclusions
plt.show()

# train_model.py
# Trains a simple linear regression model on the CSV and saves it.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import joblib
import os

# 1) read the data
df = pd.read_csv('social_media_large.csv')

# 2) choose the feature and target
X = df[['hours_per_day']]        # feature matrix (2D)
y = df['mental_health_score']    # target vector

# 3) split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4) scale feature (not strictly required for linear regression, but nice)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5) create and train the model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# 6) check performance
r2_train = model.score(X_train_scaled, y_train)
r2_test = model.score(X_test_scaled, y_test)
print(f"R2 train: {r2_train:.4f}, R2 test: {r2_test:.4f}")

# 7) save model and scaler
os.makedirs('model', exist_ok=True)
joblib.dump(model, 'model/linear_model.joblib')
joblib.dump(scaler, 'model/scaler.joblib')
print("Saved model and scaler to model/")

# 1. Bring in the tools we need
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib # Tool to save the finished model
import numpy as np

# 2. Load the data we made
print("➡️ Loading data...")
data = pd.read_csv('project_data.csv')

# 3. Separate the 'Input' (X) and the 'Answer' (Y)
# We need to reshape X because the model expects it this way
X = data['Social_Media_Usage_Hours'].values.reshape(-1, 1)
Y = data['Mental_Health_Score'].values

# 4. Split the data: A 'Training' part to teach and a 'Test' part to check
# train_test_split takes 80% for training, 20% for testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 5. Choose the Simple Linear Regression 'Teacher'
model = LinearRegression()

# 6. Teach the model! (This is where the magic happens)
print("🧠 Training the Linear Regression model...")
model.fit(X_train, Y_train)

# 7. Check how good the model is (R-squared score)
score = model.score(X_test, Y_test)
print(f"⭐ Model trained! Accuracy (R-squared) on test data: {score:.2f}")

# 8. Get the key findings (the slope and intercept)
slope = model.coef_[0]
intercept = model.intercept_
print(f"📉 Equation: Mental_Health_Score = ({slope:.2f} * Hours) + {intercept:.2f}")

# 9. **The Deployment Step!** Save the trained model so we can use it later
# We save it to a file named 'linear_model.joblib'
joblib.dump(model, 'linear_model.joblib')
print("✅ Model saved for deployment as 'linear_model.joblib'")

# 10. A small test prediction: What if someone uses social media for 4 hours?
hours_to_predict = 4.0
prediction = model.predict(np.array([[hours_to_predict]]))
print(f"\n🔮 Prediction for {hours_to_predict} hours of usage: Score is {prediction[0]:.2f}")
import pandas as pd
import numpy as np

# We want 10,000 rows of data (a "large dataset")
NUM_ROWS = 10000

# Social Media Usage (X): Hours per day, from 0 to 8
# np.random.uniform is like rolling a 10000-sided die
X_usage = np.random.uniform(0, 8, size=NUM_ROWS)

# Mental Health Score (Y): A score from 0 to 10 (10 is best)
# We make it dependent on X (usage). The negative sign means MORE usage = LOWER score.
# The + 10 is the base score. The np.random.randn adds a little "messiness" (realism)
Y_score = 10 - 1.5 * X_usage + np.random.randn(NUM_ROWS) * 1.5

# Make sure the scores don't go below 0
Y_score[Y_score < 0] = 0

# Put the data into a nice table (DataFrame)
data = pd.DataFrame({
    'Social_Media_Usage_Hours': X_usage,
    'Mental_Health_Score': Y_score
})

# Save the table into a file called 'project_data.csv'
data.to_csv('project_data.csv', index=False)

print(f"✅ Generated {NUM_ROWS} data points and saved to project_data.csv")
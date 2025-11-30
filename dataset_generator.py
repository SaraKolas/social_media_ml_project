# dataset_generator.py
# Creates a big CSV dataset that simulates social media usage vs mental health.

import numpy as np
import pandas as pd

# 1) how many rows? big number = large dataset
N = 1_000_000  # one million rows; change to 2_000_000 if you want bigger

# 2) hours on social media per day (between 0 and 12 hours)
np.random.seed(42)
hours = np.random.exponential(scale=1.8, size=N)  # many small values, some big
hours = np.clip(hours, 0, 12)  # limit to 12 hours max

# 3) true relationship: mental_health_score = base + slope * hours + noise
#    we'll make higher hours slightly increase the score (worse mental health)
base = 5.0            # baseline score
slope = 0.9           # how strongly hours affect mental health
noise = np.random.normal(loc=0.0, scale=3.0, size=N)  # random noise

mental_health = base + slope * hours + noise

# 4) optional realistic extra features (you can ignore for simple regression)
age = np.random.randint(13, 70, size=N)
gender = np.random.choice(['F','M','Other'], size=N, p=[0.48,0.48,0.04])

# 5) assemble dataframe
df = pd.DataFrame({
    'hours_per_day': hours,
    'age': age,
    'gender': gender,
    'mental_health_score': mental_health
})

# 6) save to CSV
df.to_csv('social_media_large.csv', index=False)
print("Wrote social_media_large.csv with", len(df), "rows")

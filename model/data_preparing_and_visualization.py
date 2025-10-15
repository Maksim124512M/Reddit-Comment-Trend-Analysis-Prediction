import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === Load dataset ===
df = pd.read_csv('the-reddit-dataset-dataset-comments.csv')

# === Handle missing values ===
df['sentiment'] = df['sentiment'].fillna(df['sentiment'].mean())
df['body'] = df['body'].fillna(df['body'].mode()[0])

# === Convert timestamp ===
df['created_utc'] = pd.to_datetime(df['created_utc'], unit='s')

# === Aggregate by year ===
yearly_counts = df.groupby(df['created_utc'].dt.year).size()
X = np.array(yearly_counts.index).reshape(-1, 1)  # 2D array for sklearn
y = yearly_counts.values

# Visualization
plt.style.use('ggplot')
plt.figure(figsize=(10,6))
plt.bar(X.flatten(), y)

plt.xlabel('Year')
plt.ylabel('Number of comments')
plt.title('Reddit Comment Activity by Year')

for i, val in enumerate(y):
    plt.text(i, val, str(val), ha='center', va='bottom')

plt.show()
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

plt.style.use('ggplot')    # Use the 'ggplot' style for plots

df = pd.read_csv('the-reddit-dataset-dataset-comments.csv')    # Load and prepare the dataset

# Fill missing values:
# - replace missing sentiment with the mean
# - replace missing body with the mode
df = df.fillna({
    'sentiment': df['sentiment'].mean(),
    'body': df['body'].mode()[0]})

df['created_utc'] = pd.to_datetime(df['created_utc'], unit='s')    # Convert the Unix timestamp to datetime

time_stats = df.groupby(df['created_utc'].dt.to_period('Y')).size()    # Group posts by year and count them

# Prepare X (years) and Y (number of comments) for plotting
X = time_stats.index.astype(str)
y = time_stats.values.astype(int)

# Create a bar plot
plt.bar(X, y)
plt.xlabel('Years')
plt.ylabel('Comments')

# Add exact numbers above each bar
for i, val in enumerate(y):
    plt.text(i, val, str(val), ha='center', va='bottom')

plt.show()
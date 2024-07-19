import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from datetime import datetime, timedelta

# Generate sample tweets with predefined sentiments
tweets = [
    "I love this product! It's amazing and works perfectly.",
    "This is the worst service I have ever experienced.",
    "I'm very happy with my purchase. Great quality!",
    "Terrible experience, would not recommend.",
    "It's okay, not the best but not the worst.",
    "Absolutely fantastic! Highly recommend it.",
    "Not satisfied with the product. It broke after a week.",
    "Best purchase I've made this year. Excellent value.",
    "Disappointed with the service. Expected better.",
    "It's fine. Does the job but nothing special."
]

# Define sentiment scores (TextBlob polarity for simplicity)
sentiments = [TextBlob(tweet).sentiment.polarity for tweet in tweets]

# Simulate dates for the tweets
base_date = datetime.today()
dates = [base_date - timedelta(days=i) for i in range(len(tweets))]

# Create a DataFrame
data = {
    'text': tweets,
    'date': dates,
    'polarity': sentiments
}
df = pd.DataFrame(data)

# Ensure date sorting
df = df.sort_values(by='date')

# Visualization
plt.figure(figsize=(10,6))
sns.lineplot(data=df, x='date', y='polarity', marker='o')
plt.title('Synthetic Daily Sentiment Polarity Over Time')
plt.xlabel('Date')
plt.ylabel('Sentiment Polarity')
plt.show()

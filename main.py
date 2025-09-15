import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
# Download VADER lexicon if not already present
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')
# --- 1. Synthetic Data Generation ---
np.random.seed(42) # for reproducibility
num_movies = 100
data = {
    'MovieTitle': [f'Movie {i}' for i in range(1, num_movies + 1)],
    'EarlyReviews': [f"This movie is {np.random.choice(['amazing', 'good', 'okay', 'bad', 'terrible'])}." for _ in range(num_movies)],
    'BoxOfficeWeek1': np.random.randint(100000, 10000000, size=num_movies) # Box office revenue in the first week
}
df = pd.DataFrame(data)
# --- 2. Sentiment Analysis ---
analyzer = SentimentIntensityAnalyzer()
df['SentimentScores'] = df['EarlyReviews'].apply(lambda review: analyzer.polarity_scores(review)['compound'])
# --- 3. Data Cleaning (Not strictly needed for synthetic data, but good practice) ---
# In real-world scenarios, you'd handle missing values and outliers here.  For this example, we skip this step.
# --- 4. Analysis ---
# Simple correlation analysis between sentiment and box office revenue
correlation = df['SentimentScores'].corr(df['BoxOfficeWeek1'])
print(f"Correlation between Sentiment Score and Box Office Revenue: {correlation}")
# --- 5. Visualization ---
plt.figure(figsize=(10, 6))
sns.regplot(x='SentimentScores', y='BoxOfficeWeek1', data=df)
plt.title('Sentiment Score vs. Box Office Revenue (Week 1)')
plt.xlabel('Sentiment Score')
plt.ylabel('Box Office Revenue (Week 1)')
plt.grid(True)
plt.tight_layout()
# Save the plot to a file
output_filename = 'sentiment_boxoffice.png'
plt.savefig(output_filename)
print(f"Plot saved to {output_filename}")
#Further analysis could involve building a predictive model (linear regression, etc.) using the sentiment scores to predict box office revenue.  This is omitted for brevity.
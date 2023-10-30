from flask import Flask, render_template, request, jsonify
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from collections import defaultdict
import seaborn as sns
from matplotlib import pyplot as plt

app = Flask(__name__)

# Load your data and perform sentiment analysis
df = pd.read_csv("movies_recc.csv", encoding='unicode_escape')

import nltk

nltk.download('all')

# Perform sentiment analysis and create vaders DataFrame
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()

movie_sentiments = defaultdict(list)
res = {}
for i, row in df.iterrows():
    review = row['Reviews']
    myid = row['ID']
    genre = row['Genres']
    movie_title = row['Movie']
    res[myid] = sid.polarity_scores(review)

    sentiment = sid.polarity_scores(review)['compound']
    movie_sentiments[movie_title].append(sentiment)

vaders = pd.DataFrame(res).T
vaders = vaders.reset_index().rename(columns={'index': 'ID'})
vaders = vaders.merge(df, how='left')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/recommend', methods=['POST'])
def recommend_movies():
    user_genre = request.form['genre'].strip().lower()

    filtered_movies = vaders[(vaders['Genres'].str.lower().str.contains(user_genre)) & (vaders['compound'] > 0)]
    filtered_movies = filtered_movies.sort_values(by='compound', ascending=False)

    if filtered_movies.empty:
        result = {"message": f"No movies found in the {user_genre.capitalize()} genre with positive sentiment."}
    else:
        top_movies = []
        for i, row in filtered_movies.head(3).iterrows():
            top_movies.append(f"{i + 1}. Movie: {row['Movie']}, Compound Sentiment: {row['compound']:.2f}")
        result = {"message": f"Top 3 Recommended {user_genre.capitalize()} Movies with Positive Sentiment:",
                  "movies": top_movies}

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)

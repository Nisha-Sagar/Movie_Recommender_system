# Movie Recommender system Based on Genre

This project is a movie recommender system that suggests movies to users based on their preferred movie genre. The system utilizes sentiment analysis to identify movies with positive reviews within the user's chosen genre.

Project Overview
The movie recommender system works as follows:

1. It starts by loading a dataset of movies and their associated reviews. The dataset is stored in a CSV file named "movies_recc.csv."
2. Sentiment analysis is performed on the reviews of the movies. The sentiment analysis uses the VADER (Valence Aware Dictionary and sentiment Reasoner) sentiment analysis tool, which is specifically designed for text sentiment analysis.
3. The sentiment analysis classifies reviews into three categories: positive, neutral, and negative based on the sentiment compound score.
4. Users are prompted to enter their preferred movie genre.
5. The system then filters movies by the user's preferred genre and positive sentiment (compound score > 0).
6. The filtered movies are sorted in descending order of their compound sentiment scores.
7. The system recommends the top 3 movies with the most positive reviews in the user's preferred genre.
8. Users can access the system through a web interface provided by Flask.

Prerequisites
Before running the project, ensure you have the following libraries installed:
pandas for data manipulation and analysis
numpy for numerical operations
seaborn for data visualization
nltk for natural language processing tasks, including tokenization
vaderSentiment for sentiment analysis
sklearn for machine learning tasks

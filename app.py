# Importing essential libraries and modules

from flask import Flask, render_template, request, Markup
import numpy as np
import pandas as pd

import pickle
import pandas as pd
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
import numpy as np


sm_df=pd.read_csv("tempdb.csv")

reader = Reader()
ratings = pd.read_csv('data/ratings_small.csv')

data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)


svd = SVD()
cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=5)

trainset = data.build_full_trainset()
svd.fit(trainset)

count = CountVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
count_matrix = count.fit_transform(sm_df['soup'])

cosine_sim = cosine_similarity(count_matrix, count_matrix)

sm_df = sm_df.reset_index()
titles = sm_df['title']
indices = pd.Series(sm_df.index, index=sm_df['title'])

print("working working working")

def convert_int(x):
    try:
        return int(x)
    except:
        return np.nan

id_map = pd.read_csv('data/links_small.csv')[['movieId', 'tmdbId']]
id_map['tmdbId'] = id_map['tmdbId'].apply(convert_int)
id_map.columns = ['movieId', 'id']
id_map = id_map.merge(sm_df[['title', 'id']], on='id').set_index('title')
indices_map = id_map.set_index('id')

def hybrid(userId, title):
    idx = indices[title]
    tmdbId = id_map.loc[title]['id']

    movie_id = id_map.loc[title]['movieId']

    sim_scores = list(enumerate(cosine_sim[int(idx)]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:26]
    movie_indices = [i[0] for i in sim_scores]

    movies = sm_df.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'year', 'id','genres']]
    movies['est'] = movies['id'].apply(lambda x: svd.predict(userId, indices_map.loc[x]['movieId']).est)
    movies = movies.sort_values('est', ascending=False)
    return movies.head(10)

print("working working working")







app = Flask(__name__)

# render home page


@ app.route('/')
def home():
    title = 'Movie Recommendation system using machine learning'
    return render_template('index.html', title=title)

# render recommendation form page
@ app.route('/', methods=['POST'])
def home_submit():

    user_id = int(request.form['user_id'])
    movie_name = str(request.form['movie_name'])
    dataget=hybrid(user_id, movie_name)
    title = 'Movie Recommendation system using machine learning'
    names=dataget['title'].values
    rating=dataget['est'].values
    gener=dataget['genres'].values
    final_prediction=names,rating,gener
    for i in range(0,len(final_prediction[0])):
        final_prediction[1][i]=round(final_prediction[1][i],1)
        txt=final_prediction[2][i]
        final_prediction[2][i]=str(txt.replace("'", " "))[2:-2]
    return render_template('movie-result.html', prediction=final_prediction, title=title)

@ app.route('/popular_movies.html')
def popular_movies():
    return render_template('popular_movies.html')


if __name__ == '__main__':
    app.run(debug=True)


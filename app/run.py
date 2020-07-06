<<<<<<< HEAD
<<<<<<< HEAD
# imports

from collections import Counter
import json, plotly
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from flask import Flask
from flask import render_template, request, jsonify
import numpy as np
import operator
from plotly.graph_objs import Bar
from pprint import pprint
import re
from sklearn.externals import joblib
from sqlalchemy import create_engine

# initializing Flask app
app = Flask(__name__)

def tokenize(text):
    """
    Tokenizes text data
    Args:
    text str: Messages as text data
    Returns:
    # clean_tokens list: Processed text after normalizing, tokenizing and lemmatizing
    words list: Processed text after normalizing, tokenizing and lemmatizing
    """
    # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    words = word_tokenize(text)
    
    # remove stop words
    stopwords_ = stopwords.words("english")
    words = [word for word in words if word not in stopwords_]
    
    # extract root form of words
    words = [WordNetLemmatizer().lemmatize(word, pos='v') for word in words]

    return words

engine = create_engine('sqlite:///Database.db')
df = pd.read_sql_table('Data', con=engine)
=======
=======
>>>>>>> e2d648ed519daa918cabf5075d800fdc912f5b43
import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///project.db')
df = pd.read_sql_table('clean_Data', engine)
<<<<<<< HEAD
>>>>>>> e2d648ed519daa918cabf5075d800fdc912f5b43
=======
>>>>>>> e2d648ed519daa918cabf5075d800fdc912f5b43

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
<<<<<<< HEAD
<<<<<<< HEAD
    
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message'] # message count based\
                                                          # on genre
    genre_names = list(genre_counts.index)                # genre names
    cat_p = df[df.columns[4:]].sum()/len(df)              # proportion based on\
                                                          # categories
    cat_p = cat_p.sort_values(ascending = False)          # largest bar will be\
                                                          # on left
    cats = list(cat_p.index)                              # category names

    words_with_repetition=[]                              # will contain all\
                                                          # words words with\
                                                          # repetition
    for text in df['message'].values:
        tokenized_ = tokenize(text)
        words_with_repetition.extend(tokenized_)

    word_count_dict = Counter(words_with_repetition)      # dictionary\
                                                          # containing word\
                                                          # count for all words
    
    sorted_word_count_dict = dict(sorted(word_count_dict.items(),
                                         key=operator.itemgetter(1),
                                         reverse=True))   # sort dictionary by\
                                                          # values
    top, top_10 =0, {}

    for k,v in sorted_word_count_dict.items():
        top_10[k]=v
        top+=1
        if top==10:
            break
    words=list(top_10.keys())
    pprint(words)
    count_props=100*np.array(list(top_10.values()))/df.shape[0]
    # create visuals
    figures = [
=======
=======
>>>>>>> e2d648ed519daa918cabf5075d800fdc912f5b43

    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    category_counts = df.drop(['id','message','original','genre'], axis=1).sum()
    category_names = list(category_counts.index)

    word_series = pd.Series(' '.join(df['message']).lower().split())
    top_words = word_series[~word_series.isin(stopwords.words("english"))].value_counts()[:5]
    top_words_names = list(top_words.index)


    # create visuals
    graphs = [
<<<<<<< HEAD
>>>>>>> e2d648ed519daa918cabf5075d800fdc912f5b43
=======
>>>>>>> e2d648ed519daa918cabf5075d800fdc912f5b43
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
<<<<<<< HEAD
<<<<<<< HEAD
                    x=cats,
                    y=cat_p
=======
                    x=category_names,
                    y=category_counts
>>>>>>> e2d648ed519daa918cabf5075d800fdc912f5b43
=======
                    x=category_names,
                    y=category_counts
>>>>>>> e2d648ed519daa918cabf5075d800fdc912f5b43
                )
            ],

            'layout': {
<<<<<<< HEAD
<<<<<<< HEAD
                'title': 'Proportion of Messages <br> by Category',
                'yaxis': {
                    'title': "Proportion",
                    'automargin':True
                },
                'xaxis': {
                    'title': "Category",
                    'tickangle': -40,
                    'automargin':True
=======
=======
>>>>>>> e2d648ed519daa918cabf5075d800fdc912f5b43
                'title': 'Distribution of Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
<<<<<<< HEAD
>>>>>>> e2d648ed519daa918cabf5075d800fdc912f5b43
=======
>>>>>>> e2d648ed519daa918cabf5075d800fdc912f5b43
                }
            }
        },
        {
            'data': [
                Bar(
<<<<<<< HEAD
<<<<<<< HEAD
                    x=words,
                    y=count_props
=======
                    x=top_words_names,
                    y=top_words
>>>>>>> e2d648ed519daa918cabf5075d800fdc912f5b43
=======
                    x=top_words_names,
                    y=top_words
>>>>>>> e2d648ed519daa918cabf5075d800fdc912f5b43
                )
            ],

            'layout': {
<<<<<<< HEAD
<<<<<<< HEAD
                'title': 'Frequency of top 10 words <br> as percentage',
                'yaxis': {
                    'title': 'Occurrence<br>(Out of 100)',
                    'automargin': True
                },
                'xaxis': {
                    'title': 'Top 10 words',
                    'automargin': True
=======
=======
>>>>>>> e2d648ed519daa918cabf5075d800fdc912f5b43
                'title': 'Most Frequent Words',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Words"
<<<<<<< HEAD
>>>>>>> e2d648ed519daa918cabf5075d800fdc912f5b43
=======
>>>>>>> e2d648ed519daa918cabf5075d800fdc912f5b43
                }
            }
        }
    ]
<<<<<<< HEAD
<<<<<<< HEAD
    
    # encode plotly graphs in JSON
    ids = ["figure-{}".format(i) for i, _ in enumerate(figures)]
    figuresJSON = json.dumps(figures, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly figures
    return render_template('master.html', ids=ids, figuresJSON=figuresJSON, data_set=df)

# web page that handles user query and displays model results
@app.route('/go')

def go():

    # save user input in query
    query = request.args.get('query', '') 
=======
=======
>>>>>>> e2d648ed519daa918cabf5075d800fdc912f5b43

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')
<<<<<<< HEAD
>>>>>>> e2d648ed519daa918cabf5075d800fdc912f5b43
=======
>>>>>>> e2d648ed519daa918cabf5075d800fdc912f5b43

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

<<<<<<< HEAD
<<<<<<< HEAD
    # This will render the go.html Please see that file. 
    return render_template('go.html',
                            query=query,
                            classification_result=classification_results
                          )
=======
=======
>>>>>>> e2d648ed519daa918cabf5075d800fdc912f5b43
    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )
<<<<<<< HEAD
>>>>>>> e2d648ed519daa918cabf5075d800fdc912f5b43
=======
>>>>>>> e2d648ed519daa918cabf5075d800fdc912f5b43


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
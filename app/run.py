import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine
from collections import Counter
from nltk.tokenize import RegexpTokenizer

app = Flask(__name__)


def tokenize(text):
    # tokens = word_tokenize(text)
    tokenizer = RegexpTokenizer(r'\w+')
    # remove punctuation from sentence
    tokens = tokenizer.tokenize(text)
    lemmatizer = WordNetLemmatizer()
    stop_words = stopwords.words('english')
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        if clean_tok not in stop_words:
            clean_tokens.append(clean_tok)

    return clean_tokens


# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    category_labels = list(df.columns[4:].values)
    category_count = []
    for label in category_labels:
        category_count.append(df[label].sum())

    related_all_words = []
    for index, row in enumerate(df):
        if df.iloc[index]['related'] == 1:
            related_all_words.extend(tokenize(df.iloc[index]['message']))
    word_counts = Counter(related_all_words)
    top_words = []
    top_word_counts = []
    for key, value in word_counts.most_common(5):
        top_words.append(key)
        top_word_counts.append(value)

    print(top_words)




    # create visuals
    graphs = [
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
                    x=category_labels,
                    y=category_count
                )
            ],

            'layout': {
                'title': 'Counts of Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=top_words,
                    y=top_word_counts
                )
            ],

            'layout': {
                'title': 'Counts of Top 5 words from the \'related\' category',
                'yaxis': {
                    'title': "Word Count"
                },
                'xaxis': {
                    'title': "Words"
                }
            }
        }
    ]

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

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
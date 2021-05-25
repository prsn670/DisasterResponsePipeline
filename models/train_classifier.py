import sys
import pickle
import warnings

from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import LinearSVC
from sqlalchemy import create_engine, Table, MetaData
import pandas as pd
import nltk
from nltk import word_tokenize, WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
stop_words = stopwords.words('english')
tfidf_vectorizer = TfidfVectorizer(use_idf=True, lowercase=True, analyzer='word', stop_words=stop_words)
X_messages_train = X_messages_test = y_categories_train = y_categories_test = None


def create_test_data(db_location='data/DisasterResponse.db'):
    """
    Splits data into training and testing groups.
    :return:
    """


    engine = create_engine('sqlite:///' + db_location)
    table_name = db_location.split('.')[0].replace('data/', '')
    df = pd.read_sql_table(table_name, engine)

    global X_messages_train
    global X_messages_test
    global y_categories_train
    global y_categories_test
    X_messages = df.message.values
    y_categories = df.drop(columns=['id', 'original', 'message', 'genre'])
    X_messages_train, X_messages_test, y_categories_train, y_categories_test = train_test_split(
        X_messages,
        y_categories,
        test_size=0.3)
    return y_categories.columns

# tokenize, lemmatize, and normalize text.
def tokenize(message_data):
    """
    Takes in a message, tokenizes, lemmatizes, and then normalizes the tokens by making all characters in each word.
    If part of the training pipeline, also applies TF-IDF
        lowercase
    :param process:
    :param message:
    :return:
    """
    normalized_words = []
    stop_words = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()
    for message in message_data:
        words = word_tokenize(message)
        for word in words:
            if word not in stop_words:
                normalized_words.append(lemmatizer.lemmatize(word).lower())
    return normalized_words


def build_model():
    """
    builds model using pipeline, Linear SVC, and GridSearch
    :return:
    """
    disaster_pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(OneVsRestClassifier(LinearSVC())))
    ])
    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'vect__max_df': (0.5, 0.75, 1.0),
        # 'vect__max_features': (None, 5000, 10000),
        # 'tfidf__use_idf': (True, False),

    }
    cv = GridSearchCV(disaster_pipeline, param_grid=parameters)
    return cv

def print_report(test_values, prediction, labels):
    """
    Prints report showing precision, recall f1-score support, and accuracy for each category label.
    :param test_values:
    :param prediction:
    :param labels:
    :return:
    """
    print(classification_report(test_values, prediction, target_names=labels))
    accuracy = (y_pred == y_categories_test).mean()
    print(f'Accuracy: \n{accuracy}')

def create_pickle_file(model, filepath='models/classifier.pkl'):
    """
    saves the resulting model as a pkl file
    :param model:
    :param filepath:
    :return:
    """
    pickle.dump(model, open(filepath, 'wb'))


if __name__ == '__main__':
    # create data that will be used for testing
    try:
        labels = create_test_data(sys.argv[1])
    except IndexError:
        warnings.warn("The path to the db was not entered. Will read file from default location.")
        labels = create_test_data()

    # build and fit our model
    model = build_model()
    model.fit(X_messages_train, y_categories_train)

    # predict categories on testing data
    y_pred = model.predict(X_messages_test)

    # print report based on predictions
    print_report(y_categories_test.values, y_pred, labels)

    # save model as pkl file
    try:
        create_pickle_file(model, sys.argv[2])
    except IndexError:
        warnings.warn("Path to pickle file not entered. Using default location")
        create_pickle_file(model)


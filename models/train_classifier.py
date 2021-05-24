import sys
import pickle
import warnings

from sklearn.metrics import confusion_matrix
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import LinearSVC
from sqlalchemy import create_engine, Table, MetaData
import pandas as pd
from nltk import word_tokenize, WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

stop_words = stopwords.words('english')
tfidf_vectorizer = TfidfVectorizer(use_idf=True, lowercase=True, analyzer='word', stop_words=stop_words)
X_messages_train = X_messages_test = y_categories_train = y_categories_test = None


def create_test_data(db_location='data/DisasterResponse.db'):
    """
    Splits data into training and testing groups.
    :return:
    """


    engine = create_engine('sqlite:///' + db_location)
    metadata = MetaData()
    table_name = db_location.split('.')[0].replace('data/', '')
    # disaster_data = Table(table_name, metadata, autoload=True, autoload_with=engine)
    df = pd.read_sql_table(table_name, engine)

    global X_messages_train
    global X_messages_test
    global y_categories_train
    global y_categories_test
    X_messages = df.message.values
    y_categories = df.drop(columns=['id', 'original', 'message', 'genre'])
    X_messages_train, X_messages_test, y_categories_train, y_categories_test = train_test_split(
        X_messages,
        y_categories.to_numpy(),
        test_size=.10)
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
    # tfidf_vectorizer = TfidfVectorizer(use_idf=True, lowercase=True, analyzer='word', stop_words=stop_words)
    # if process == 'training':
    # idf = tfidf_vectorizer.fit_transform(message_data)
    # idf_dataframe = pd.DataFrame(idf[0].T.todense(), index=tfidf_vectorizer.get_feature_names(), columns=["TF-IDF"])
    # # elif process == 'testing':
    for message in message_data:
        words = word_tokenize(message)
        for word in words:
            if word not in stop_words:
                normalized_words.append(lemmatizer.lemmatize(word).lower())
    return normalized_words


def build_model():
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
    print(classification_report(test_values, prediction, target_names=labels))
    accuracy = (y_pred == y_categories_test).mean()
    print_report(f'Accuracy: {accuracy}')

def create_pickle_file(model, filepath='models/classifier.pkl'):
    pickle.dump(model, open(filepath, 'w'))


if __name__ == '__main__':
    try:
        labels = create_test_data(sys.argv[1])
    except IndexError:
        warnings.warn("The path to the db was not entered. Will read file from default location.")
        labels = create_test_data()

    model = build_model()
    model.fit(X_messages_train, y_categories_train)
    y_pred = model.predict(X_messages_test)

    print_report(y_categories_test.values, y_pred, labels)
    try:
        create_pickle_file(model, sys.argv[2])
    except IndexError:
        warnings.warn("Path to pickle file not entered. Using default location")
        create_pickle_file(model)


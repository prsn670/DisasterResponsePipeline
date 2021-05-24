from sklearn.metrics import confusion_matrix
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import SVC, LinearSVC
from sqlalchemy import create_engine, Table, MetaData
import pandas as pd
from nltk import word_tokenize, WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
import numpy as np

stop_words = stopwords.words('english')
tfidf_vectorizer = TfidfVectorizer(use_idf=True, lowercase=True, analyzer='word', stop_words=stop_words)
X_messages_train = X_messages_test = y_categories_train = y_categories_test = None


def create_test_data():
    """
    Splits data into training and testing groups.
    :return:
    """
    db_location = 'data/DisasterResponse.db'

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
        test_size=.30,
        random_state=42)


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
    dataset = [
        "I enjoy reading about Machine Learning and Machine Learning is my PhD subject",
        "I would enjoy a walk in the park",
        "I was reading in the library"
    ]
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


if __name__ == '__main__':
    create_test_data()
    model = build_model()
    model.fit(X_messages_train, y_categories_train)
    y_pred = model.predict(X_messages_test)

    labels = np.unique(y_pred)
    confusion_mat = confusion_matrix(y_categories_test, y_pred, labels=labels)
    accuracy = (y_pred == y_categories_test).mean()


    print("Labels:", labels)
    print("Confusion Matrix:\n", confusion_mat)
    print("Accuracy:", accuracy)
    print("\nBest Parameters:", model.best_params_)
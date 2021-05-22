from sqlalchemy import create_engine, Table, MetaData
import pandas as pd
from nltk import word_tokenize, WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


class TrainModel:

    def __init__(self, db_location='data/DisasterResponse.db'):
        self._engine = create_engine('sqlite:///' + db_location)
        metadata = MetaData()
        table_name = db_location.split('.')[0].replace('data/', '')
        self.disaster_data = Table(table_name, metadata, autoload=True, autoload_with=self._engine)
        self.df = pd.read_sql_table(table_name, self._engine)
        self.X_messages_train = self.X_messages_test = self.y_categories_train = self.y_categories_test = None

    def create_test_data(self):
        """
        Splits data into training and testing groups.
        :return:
        """
        X_messages = self.df['message']
        y_categories = self.df.drop(columns=['id', 'original'])
        self.X_messages_train, self.X_messages_test, self.y_categories_train, self.y_categories_test = train_test_split(
            X_messages,
            y_categories,
            test_size=.30,
            random_state=42)
        return X_messages, y_categories

    # tokenize, lemmatize, and normalize text.
    def tokenize(self, message, process):
        """
        Takes in a message, tokenizes, lemmatizes, and then normalizes the tokens by making all characters in each word
            lowercase
        :param message:
        :return:
        """
        stop_words = stopwords.words('english')
        words = word_tokenize(message)
        lemmatizer = WordNetLemmatizer()
        normalized_words = []
        for word in words:
            if (word not in stop_words):
                normalized_words.append(lemmatizer.lemmatize(word).lower())
        return normalized_words
        # if (process == 'training'):
        #     tfidf_vectorizer = TfidfVectorizer()
        #     return tfidf_vectorizer.fit_transform(normalized_words)

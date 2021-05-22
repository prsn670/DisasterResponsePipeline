from sqlalchemy import create_engine, Table, MetaData
import pandas as pd
from nltk import word_tokenize, WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import time


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
        y_categories = self.df.drop(columns=['id', 'original', 'message', 'genre'])
        self.X_messages_train, self.X_messages_test, self.y_categories_train, self.y_categories_test = train_test_split(
            X_messages,
            y_categories,
            test_size=.30,
            random_state=42)
        return X_messages, y_categories

    # tokenize, lemmatize, and normalize text.
    def tokenize(self, message_data, process):
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
        tfidf_vectorizer = TfidfVectorizer(use_idf=True, lowercase=True, analyzer='word', stop_words=stop_words)
        if process == 'training':
            self.idf = tfidf_vectorizer.fit_transform(message_data)
            self.idf_dataframe = pd.DataFrame(self.idf[0].T.todense(), index=tfidf_vectorizer.get_feature_names(), columns=["TF-IDF"])
        elif process == 'testing':
            for message in message_data:
                words = word_tokenize(message)
                for word in words:
                    if word not in stop_words:
                        normalized_words.append(lemmatizer.lemmatize(word).lower())



    def pipeline(self):
        dataset = [
            "I enjoy reading about Machine Learning and Machine Learning is my PhD subject",
            "I would enjoy a walk in the park",
            "I was reading in the library"
        ]
        disaster_pipeline = Pipeline([('tokenize', self.tokenize(self.X_messages_train, 'training'))])
        return self.idf_dataframe, self.idf

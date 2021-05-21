from sqlalchemy import create_engine, Table, MetaData
import pandas as pd


class TrainModel:

    def __init__(self, db_location='data/DisasterResponse.db'):
        self._engine = create_engine('sqlite:///' + db_location)
        metadata = MetaData()
        table_name = db_location.split('.')[0].replace('data/', '')
        self.disaster_data = Table(table_name, metadata, autoload=True, autoload_with=self._engine)
        self.df = pd.read_sql_table(table_name, self._engine)

    # TODO: think of better method name
    def create_test_data(self):
        X_messages = self.df['message']
        y_categories = self.df.drop(columns=['id', 'original'])
        return X_messages, y_categories

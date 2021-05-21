from sqlalchemy import create_engine, Table, MetaData


class TrainModel:

    def __init__(self):
        engine = create_engine('sqlite:///data/DisasterMessage.db')
        metadata = MetaData()
        self.disaster_data = Table('DisasterMessage', metadata, autoload=True, autoload_with=engine)


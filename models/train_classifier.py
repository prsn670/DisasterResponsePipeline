import os
from TrainModel import TrainModel

if __name__ == "__main__":
    data = TrainModel()
    print(data.disaster_data.columns.keys())

import sys
import pandas as pd
from TrainModel import TrainModel

if __name__ == "__main__":
    try:
        data = TrainModel(sys.argv[1])
    except IndexError:
        data = TrainModel()

    X = data.create_test_data()
    # data.tokenize(data.X_messages_train[0], "training")
    p = data.pipeline()

    p[0].sort_values(by=['TF-IDF'], inplace=True, ascending=False)




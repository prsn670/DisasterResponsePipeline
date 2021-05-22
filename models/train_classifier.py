import sys
from TrainModel import TrainModel

if __name__ == "__main__":
    try:
        data = TrainModel(sys.argv[1])
    except IndexError:
        data = TrainModel()

    X = data.create_test_data()
    idf = data.tokenize(data.X_messages_train[0], "training")

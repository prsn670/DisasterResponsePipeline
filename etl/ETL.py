import pandas as pd


class ETL:

    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.df_merged = None
        self.df_messages = None
        self.df_categories = None

    def load(self, messages_csv="data/messages.csv", categories_csv="data/categories.csv"):
        """
        Reads csv files and puts them in a dataframe.
        :param messages_csv: file location of messages
        :param categories_csv:
        :return:
        """
        self.df_messages = pd.read_csv(self.root_dir + "/" + messages_csv)
        self.df_categories = pd.read_csv(self.root_dir + "/" + categories_csv)

    def merge_df(self):
        """
        Merges df_messages and df_categories into df_merged
        :return:
        """
        self.df_merged = pd.merge(self.df_messages, self.df_categories, on=['id', 'id'])

    def get_merged_df(self):
        return self.df_merged

    # TODO Split categories, convert to 0 or 1, and replace column with new category columns
    # TODO remove duplicates
    # TODO save dataset into sqlite DB

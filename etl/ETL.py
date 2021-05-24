
import pandas as pd
import re
from sqlalchemy import create_engine

class ETL:

    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.df_merged = None
        self.df_messages = None
        self.df_categories = None
        self.engine = None
        self.table_name = None

    def read_csv(self, messages_csv="data/disaster_messages.csv", categories_csv="data/disaster_categories.csv"):
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
        self.split_category()

    def get_merged_df(self):
        return self.df_merged

    def split_category(self):
        """
        Splits the data in the categories column and creates a new columns based on the category name.
        The split values are then moved back into df_merged
        :return:
        """
        # create new category column names
        self.category_col_names = re.split('-\d;?', self.df_categories['categories'][0])
        self.category_col_names.remove('')

        # split values in category column and create new data frame
        category_series = pd.Series(self.df_merged['categories'])
        split_categories_df = category_series.str.split(pat=";", expand=True)

        # rename columns
        split_categories_df.columns = self.category_col_names
        # remove non-number characters leaving only 0 or 1
        split_categories_df.replace(to_replace='\w+-', value="", inplace=True, regex=True)
        # convert strings 0 and 1 to integers
        split_categories_df = split_categories_df.astype(int)

        # concatenate split data frame into merged data frame and drop old category column
        self.df_merged = pd.concat([self.df_merged, split_categories_df], axis=1)

    def clean_df(self):
        """
        Removes categories column from merged data frame and removes duplicates
        :return:
        """
        self.df_merged.drop(['categories'], axis=1, inplace=True)

        # remove duplicates
        self.df_merged.drop_duplicates(inplace=True)

        # remove row if any category is not a 0 or 1
        for column_name in self.category_col_names:
            try:
                self.df_merged.drop(self.df_merged[self.df_merged[column_name] < 0].index, inplace=True)
                self.df_merged.drop(self.df_merged[self.df_merged[column_name] > 1].index, inplace=True)
            except KeyError:
                continue




    def df_insert_db(self, db_location='data/DisasterResponse.db'):
        """
        inserts data frame into a SQLite database
        :return:
        """
        # insert dataset into sqlite DB
        self.engine = create_engine('sqlite:///' + db_location)
        table_name = db_location.split('.')[0].replace('data/', '')
        self.df_merged.to_sql(table_name, self.engine, index=False, if_exists='replace')

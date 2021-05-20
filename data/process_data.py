import os
from etl import ETL

if __name__ == "__main__":
    root = os.path.dirname(os.path.abspath('setup.py'))  # This is your Project Root - use with terminal
    # root = os.path.dirname('../setup.py')  # This is your Project Root - Use with pycharm
    etl_obj = ETL(root)
    etl_obj.load()
    etl_obj.merge_df()
    etl_obj.clean_df()
    etl_obj.df_insert_db()

    print(etl_obj.get_merged_df().head())
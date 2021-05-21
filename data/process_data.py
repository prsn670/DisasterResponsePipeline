import os
import sys
import warnings
import etl

if __name__ == "__main__":
    root = os.path.dirname(os.path.abspath('setup.py'))  # This is your Project Root

    etl_obj = etl.ETL(root)
    try:
        etl_obj.read_csv(sys.argv[1], sys.argv[2])
    except IndexError:
        warnings.warn("The two csv files to read from were not entered. Will read files from default location.")
        etl_obj.read_csv()

    etl_obj.merge_df()
    etl_obj.clean_df()
    try:
        etl_obj.df_insert_db(sys.argv[3])
    except IndexError:
        warnings.warn("The database location was not entered, using default value.")
        etl_obj.df_insert_db()



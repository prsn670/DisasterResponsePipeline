import os
import sys
import warnings
import etl

if __name__ == "__main__":
    root = os.path.dirname(os.path.abspath('setup.py'))  # This is your Project Root

    # initialize ETL object
    etl_obj = etl.ETL(root)
    # read in csv files
    try:
        etl_obj.read_csv(sys.argv[1], sys.argv[2])
    except IndexError:
        warnings.warn("The two csv files to read from were not entered. Will read files from default location.")
        etl_obj.read_csv()

    # merge in both data frames created when csv files read in
    etl_obj.merge_df()

    # clean the merged data frame
    etl_obj.clean_df()

    # insert merged data frame into a database
    try:
        etl_obj.df_insert_db(sys.argv[3])
    except IndexError:
        warnings.warn("The database location was not entered, using default value.")
        etl_obj.df_insert_db()

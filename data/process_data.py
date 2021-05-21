import os
import sys
import warnings
import etl

if __name__ == "__main__":
    root = os.path.dirname(os.path.abspath('setup.py'))  # This is your Project Root

    etl_obj = etl.ETL(root)
    if len(sys.argv) == 3:
        etl_obj.read_csv(sys.argv[1], sys.argv[2])
    else:
        warnings.warn("The correct number of arguments(3) were not entered. Will read files from default location.")
        etl_obj.read_csv()

    etl_obj.merge_df()
    etl_obj.clean_df()
    etl_obj.df_insert_db()

    print(sys.argv)
    print(etl_obj.get_merged_df().head())

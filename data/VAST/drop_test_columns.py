from os.path import abspath

from datasets import DatasetDict, Dataset
import pandas as pd


def main(output_test_csv="vast_test_cleaned.csv"):
    """load VAST from local CSV and upload to HuggingFace Hub"""
    train_csv = abspath("vast_train.csv")
    val_csv = abspath("vast_dev.csv")
    test_csv = abspath("vast_test.csv")

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    # take common columns
    columns = set(train_df.columns) & set(val_df.columns) & set(test_df.columns)
    columns_list = list(columns)
    train_df = train_df[columns_list]
    val_df = val_df[columns_list]
    test_df = test_df[columns_list]

    # save the new test_df
    test_df.to_csv(output_test_csv, index=False)


if __name__ == "__main__":
    main()

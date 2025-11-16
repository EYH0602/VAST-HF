from os.path import abspath

from datasets import DatasetDict, Dataset
import pandas as pd


def main():
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

    dataset = DatasetDict()
    dataset["train"] = Dataset.from_pandas(train_df)
    dataset["validation"] = Dataset.from_pandas(val_df)
    dataset["test"] = Dataset.from_pandas(test_df)

    dataset.push_to_hub("yfhe/VAST", private=False)


if __name__ == "__main__":
    main()

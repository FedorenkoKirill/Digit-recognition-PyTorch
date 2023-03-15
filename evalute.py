import pandas as pd
from fire import Fire
from sklearn.metrics import accuracy_score, confusion_matrix


def main(dataset_path: str, result_path: str):
    dataset_df = pd.read_csv(dataset_path)
    result_df = pd.read_csv(result_path)
    print(f"Accuracy: {accuracy_score(dataset_df['target'].values, result_df['target'].values)}")
    print(f"Confusion matrix:\n{confusion_matrix(dataset_df['target'].values, result_df['target'].values)}")


if __name__ == "__main__":
    Fire(main)

import os
from pathlib import Path

import pandas as pd
from fire import Fire


def dataset_dir2csv(dataset_dir: str, file_path: str):
    files_path = []
    targets = []
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            files_path.append(Path(root, file))
            targets.append(Path(root).name)

    df = pd.DataFrame(data={"file_path": files_path, "target": targets})
    df.to_csv(file_path, sep=',', index=False)


if __name__ == "__main__":
    Fire(dataset_dir2csv)

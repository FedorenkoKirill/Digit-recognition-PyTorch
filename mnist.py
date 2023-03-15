import os
from pathlib import Path

import pandas as pd
import torch
from alive_progress import alive_bar
from fire import Fire
from torch import nn, optim
from torch.utils.data import DataLoader

from model import Net
from training_hyperparams import *
from data_module import ClassifierDataset
from transforms import input_transform, get_image_by_tensor


def train(dataset_path: str, save_dir: str):
    # Get training dataset
    train_df = pd.read_csv(dataset_path)
    train_loader = DataLoader(
        dataset=ClassifierDataset(train_df, input_transform()),
        batch_size=batch_size,
        shuffle=shuffle
    )

    # Create model
    model = Net()

    # Set criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    # Training loop
    for e in range(epochs):
        running_loss = 0.0
        with alive_bar(len(train_loader), title=f"Epoch {e+1}") as bar:
            for _, data, target in train_loader:
                # Without gradient
                optimizer.zero_grad()
                # forward + backward
                outputs = model(data)

                # Calculate loss
                loss = criterion(outputs, target)
                loss.backward()
                running_loss += loss.item()

                # Update params
                optimizer.step()
                bar()
            print(f"Training loss: {running_loss / len(train_loader)}")

    # Save model
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), Path(save_dir, "model.pth"))


def inference(model_path: str, dataset_path: str, save_dir: str):
    # Get test dataset
    test_df = pd.read_csv(dataset_path)
    test_loader = DataLoader(
        dataset=ClassifierDataset(test_df, input_transform()),
        batch_size=1,
        shuffle=False
    )

    # Load model
    model = Net()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Testing loop
    with alive_bar(len(test_loader)) as bar:
        for i, data, _ in test_loader:
            # Get predict for image
            output = model(data)
            _, predicted = torch.max(output, 1)

            # Create dirs for result
            os.makedirs(Path(save_dir, str(int(predicted[0]))), exist_ok=True)

            # Save image
            get_image_by_tensor()(data.squeeze_(0)).save(Path(save_dir, str(int(predicted[0])), f"{i}.png"))
            bar()


if __name__ == "__main__":
    Fire()

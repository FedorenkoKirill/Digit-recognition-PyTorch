from PIL import Image
from torch.utils.data import Dataset


class ClassifierDataset(Dataset):
    def __init__(
            self,
            data,
            transform
    ):
        super(ClassifierDataset, self).__init__()
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, target = self.data.iloc[[idx]].values[0]
        image = Image.open(path)
        if self.transform:
            image = self.transform(image)
        return idx, image, target

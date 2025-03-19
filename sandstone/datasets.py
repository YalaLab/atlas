from torchvision import datasets
from torchvision import transforms
import os


class ImageFolderDataset(datasets.ImageFolder):
    def __init__(self, args, augmentations, split_group, data_path, **kwargs):
        if split_group == "dev":
            split_group = "val"
        transform = transforms.Compose(augmentations)
        super().__init__(os.path.join(data_path, split_group), transform=transform, **kwargs)
        self.args = args

    def __getitem__(self, index):
        item = super().__getitem__(index)
        item = {
            "x": item[0],
            "y": item[1],
        }
        return item

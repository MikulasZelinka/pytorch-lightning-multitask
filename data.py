import torch
# from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset


class SampleSentenceDataset(Dataset):
    # TODO – probably use DataModules
    # https://pytorch-lightning.readthedocs.io/en/stable/notebooks/lightning_examples/datamodules.html#Using-DataModules

    def __init__(self):
        self.data = [
            {"sentence": "The site was empty.", "sentiment": 0, "goal": 0},
            {"sentence": "I purchased shoes but they arrived damaged!", "sentiment": 0, "goal": 1},
            {"sentence": "I wanted to buy a jacket but the page kept crashing...", "sentiment": 0, "goal": 2},
            {"sentence": "Looks nice, waiting for the shop to open", "sentiment": 1, "goal": 0},
            {"sentence": "Thanks for the nice deal", "sentiment": 1, "goal": 1},
            {"sentence": "Good website but I haven't decided what to order yet.", "sentiment": 1, "goal": 2},
        ] * 100

        self.labels = {
            "sentiment": {0: "negative", 1: "positive"},
            "goal": {0: "no_goal", 1: "met", 2: "not_met"},
        }

    def __getitem__(self, index):
        item = self.data[index]
        # returns (x, y_dict)
        return item["sentence"], {"sentiment": item["sentiment"], "goal": item["goal"]}

    def __len__(self):
        return len(self.data)

import torch
from torch import nn


class TaskHead(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss = None


class Task:
    name: str = "Task"
    output_size: int
    head: nn.Module

    def build_head(self, input_size: int) -> TaskHead:
        pass


class ClassificationTask(Task):

    def __init__(self, name, labels: dict = None):
        super().__init__()
        self.name = name
        self.labels = labels
        self.output_size = len(labels)

    def build_head(self, input_size: int):
        return ClassificationHead(input_size=input_size, output_size=self.output_size)

    def __str__(self):
        return f'ClassificationTask-{self.name or self.output_size}'


class ClassificationHead(TaskHead):

    def __init__(self, input_size, output_size, class_weights=None):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.class_weights = class_weights

        self.layer = nn.Linear(input_size, output_size)

        if self.class_weights:
            self.class_weights = torch.tensor(class_weights)

        self.loss = torch.nn.CrossEntropyLoss(weight=self.class_weights)

    def forward(self, x):
        # TODO: create tensors earlier in Dataset/Dataloader (instead of in the model)
        return self.layer.forward(torch.from_numpy(x))

    def __str__(self):
        return f'ClassificationHead-{self.input_size}-{self.output_size}'

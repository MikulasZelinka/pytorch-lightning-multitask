import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from sentence_transformers import SentenceTransformer
from torch import nn
from torch.nn import Module

from task import Task, TaskHead


class MultiTaskModel(pl.LightningModule):
    """
    MTModel has an Encoder and task-specific Heads.

    Encoder:
        - the common part of the model (e.g., it encodes an input sequence into a fixed-size vector).
    Head
        - task-specific part of the model, usually just a Sequential NN. Takes input from the Encoder output.
        - is automatically built from the "tasks" list based on each Task
        - MTModel can have one or more tasks (task heads)
    """

    def __init__(self, encoder: Module, encoder_size: int, tasks: list[Task]):
        super().__init__()

        self.encoder = encoder
        self.tasks = tasks
        self.heads: nn.ModuleDict[str, TaskHead] = nn.ModuleDict({
            task.name: task.build_head(encoder_size)
            for task in tasks
        })

    def get_logger(self):
        # TODO: tokenize text in Dataset/Dataloader so that we can log the graph
        # self.example_input_array = ("Example input sentence.",)
        return TensorBoardLogger("_tb", name="MultiTaskModel", log_graph=False)

    def forward(self, x):
        if isinstance(self.encoder, SentenceTransformer):
            # SentenceTransformers use the encode() method on a list of sentences:
            x_encoded = self.encoder.encode(list(x))
            # for more granular use, see:
            # https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2#usage-huggingface-transformers
            # where we would also do tokenization, pooling etc. separately
        else:
            # otherwise, standard pytorch nn.Modules just use a direct call:
            x_encoded = self.encoder(x)

        # call each task head on the encoded "x" input
        y_pred = {}
        for task_name, task_head in self.heads.items():
            y_pred[task_name] = task_head(x_encoded)

        return x_encoded, y_pred

    def training_step(self, batch, batch_idx):
        x, y_true = batch

        # if not self.written_graph:
        #     self.logger.log_graph(self, x)
        #     self.written_graph = True

        # "x_encoded" isn't necessarily needed depending on the task(s), but we can still use it for logging etc.
        x_encoded, y_pred = self.forward(x)

        # keep track of individual losses (to show them in tensorboard, for example)
        losses = {
            task_name: task_head.loss(y_pred[task_name], y_true[task_name])
            for task_name, task_head in self.heads.items()
        }
        self.log_dict({
            f'loss_train/{task_name}': task_loss
            for task_name, task_loss in losses.items()
        })

        # losses could be aggregated arbitrarily, here we simply sum them up
        loss = sum(losses.values())
        self.log(f'loss_train/sum', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

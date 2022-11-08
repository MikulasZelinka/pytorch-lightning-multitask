from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from data import SampleSentenceDataset
from model import MultiTaskModel
from task import ClassificationTask

from sentence_transformers import SentenceTransformer

dataset = SampleSentenceDataset()
tasks = [
    ClassificationTask(name="sentiment", labels=dataset.labels["sentiment"]),
    ClassificationTask(name="goal", labels=dataset.labels["goal"]),
]
train_loader = DataLoader(dataset, batch_size=5)

encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
multi_task_model = MultiTaskModel(encoder=encoder, encoder_size=len(encoder.encode("test")), tasks=tasks)

trainer = Trainer(logger=multi_task_model.get_logger())
trainer.fit(model=multi_task_model, train_dataloaders=train_loader)

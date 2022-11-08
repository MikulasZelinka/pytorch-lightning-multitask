Proof of concept of a multi-task model
with a common encoder (e.g., (`sentence`|`image`|`...`) (`transformer`|`RNN`|`...`))
and independent task heads
trained jointly
in [Pytorch Lightning](https://github.com/Lightning-AI/lightning).

We use a pretrained [SentenceTransformer](https://github.com/UKPLab/sentence-transformers/)
as the sentence encoder in our example.
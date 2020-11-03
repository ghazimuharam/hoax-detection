import torch
import torch.nn.functional as F
from transformers import BertForSequenceClassification, BertConfig, BertTokenizer

HF_MODEL = 'indobenchmark/indobert-base-p1'
LC_MODEL = './saved_model'

idx2lbl = {0: 'Not Hoax', 1: 'Hoax'}

tokenizer = BertTokenizer.from_pretrained(HF_MODEL)
config = BertConfig.from_pretrained(LC_MODEL)
config.num_labels = 2

device = "cuda" if torch.cuda.is_available() else "cpu"

# Instantiate model
model = BertForSequenceClassification.from_pretrained(
    LC_MODEL, config=config)

model.to(device)


def make_prediction(text_narration):
    """
    Make a prediction from a string
    return logits, label
    """
    text = text_normalization(text_narration)
    subwords = tokenizer.encode(text)
    subwords = torch.LongTensor(subwords).view(1, -1).to(model.device)

    logits = model(subwords)[0]
    max_predict = torch.topk(logits, k=1, dim=-1)[1].squeeze().item()
    label = F.softmax(logits, dim=-1).squeeze()
    return logits.tolist(), label.tolist(), max_predict


def text_normalization(text_narration):
    """
    This method normalize text provided by users
    """
    text_narration = text_narration.lower()
    return text_narration

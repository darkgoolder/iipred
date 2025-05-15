import torch
from torch.utils.data import Dataset
from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification
from transformers import Trainer, TrainingArguments
from transformers import AdamW
import numpy as np
from transformers import EarlyStoppingCallback

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DebertaV2WithDropout(DebertaV2ForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.dropout = torch.nn.Dropout(0.1).to(device)

    def forward(self, input_ids, attention_mask=None, labels=None):
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        if labels is not None:
            labels = labels.to(device)

        outputs = self.deberta(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = torch.nn.BCEWithLogitsLoss().to(device)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))

        output = (logits,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float).to(device)
        }

def to_multihot(labels, num_classes=4):
    multihot = np.zeros((len(labels), num_classes))
    for i, label in enumerate(labels):
        multihot[i, label] = 1
    return multihot

def predict_style(text, model, tokenizer):
    model.eval()
    encoding = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=512,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    probabilities = torch.sigmoid(logits).cpu().numpy().flatten()
    return {
        "DeepSeek": float(probabilities[0]),
        "Mixtral": float(probabilities[1]),
        "DeepAI": float(probabilities[2]),
        "Human": float(probabilities[3])
    }
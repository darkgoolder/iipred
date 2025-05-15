import torch
from torch.utils.data import Dataset, random_split
from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification
from transformers import Trainer, TrainingArguments
from transformers import AdamW
import numpy as np
from transformers import EarlyStoppingCallback
from model import DebertaV2WithDropout, TextDataset, to_multihot, predict_style

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def load_data(file_paths, labels):
    texts = []
    labels_list = []
    for file_path, label in zip(file_paths, labels):
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
            texts.extend(lines)
            labels_list.extend([label] * len(lines))
    return texts, labels_list

class FullDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

def main():
    # Загрузка данных
    file_paths = ['data/deepsekk.txt', 'data/mistral.txt', 'data/depai.txt', 'data/chel.txt']
    labels = [0, 1, 2, 3]  # 0 - DeepSeek, 1 - Mixtral, 2 - DeepAI, 3 - human
    texts, labels_list = load_data(file_paths, labels)

    # Преобразование в multi-hot encoding
    multihot_labels = to_multihot(labels_list)

    # Создание полного датасета
    full_dataset = FullDataset(texts, multihot_labels)

    # Разделение на обучающую и валидационную выборки
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Извлечение данных из поддатасетов
    train_texts = [full_dataset.texts[i] for i in train_dataset.indices]
    train_labels = [full_dataset.labels[i] for i in train_dataset.indices]

    val_texts = [full_dataset.texts[i] for i in val_dataset.indices]
    val_labels = [full_dataset.labels[i] for i in val_dataset.indices]

    # Инициализация модели
    model = DebertaV2WithDropout.from_pretrained(
        'microsoft/deberta-v3-base',
        num_labels=4,
        problem_type="multi_label_classification"
    ).to(device)

    tokenizer = DebertaV2Tokenizer.from_pretrained('microsoft/deberta-v3-base')

    # Настройка оптимизатора
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

    # Настройка аргументов обучения
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=25,
        per_device_train_batch_size=8 if device.type == 'cuda' else 4,
        per_device_eval_batch_size=8 if device.type == 'cuda' else 4,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir='./logs',
        logging_steps=10,
        load_best_model_at_end=True,
        learning_rate=2e-5,
        weight_decay=0.01,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=device.type == 'cuda',
    )

    # Создание Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=TextDataset(train_texts, train_labels, tokenizer),
        eval_dataset=TextDataset(val_texts, val_labels, tokenizer),
        tokenizer=tokenizer,
        optimizers=(optimizer, None),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    # Обучение модели
    trainer.train()

    # Сохранение модели
    model.save_pretrained("./text_style_detector")
    tokenizer.save_pretrained("./text_style_detector")

    print("Обучение завершено. Модель сохранена.")

if __name__ == "__main__":
    main()
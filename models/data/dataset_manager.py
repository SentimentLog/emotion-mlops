import torch
from torch.utils.data import Dataset
import json
import os
from typing import List

class DatasetManager(Dataset):
    def __init__(self, path:str, tokenizer, max_length: int=128):
        self.path = path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_labels = 7

        self.encodings, self.labels = self.load_and_preprocess()

    def load_and_preprocess(self):
        data = self.load_data(self.path)

        texts = [item['발화'] for item in data]
        labels = [item['label'] for item in data]

        # tokenizer
        encodings = self.tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return encodings, labels

    def load_data(self, path:str) -> List:
        if not os.path.exists(path):
            raise FileNotFoundError(f'데이터를 찾을 수 없음 :{path}')

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return data

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['label'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)



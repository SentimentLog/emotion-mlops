from src.ModelManager import ModelManager
import os
import json

class DatasetManager:
    def __init__(self):
        self.tokenizer = ModelManager().load_tokenizer()

    def load_data(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f'{path} 데이터 없음')
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    def tokenizer_settings(self, datasets):
        return self.tokenizer(
            datasets['발화'],
            padding='max_length',
            truncation=True,
            max_length=128,
        )




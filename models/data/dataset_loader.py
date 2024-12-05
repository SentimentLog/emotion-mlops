from .dataset_manager import DatasetManager

class DatasetLoader:
    def __init__(self, train_path: str, val_path: str, tokenizer, max_length: int = 128):
        self.train_path = train_path
        self.val_path = val_path
        self.tokenizer = tokenizer
        self.max_length = max_length

    def get_datasets(self) -> dict:
        datasets = {}
        if self.train_path:
            print("훈련 데이터 로드 중...")
            train_dataset = DatasetManager(self.train_path, self.tokenizer, self.max_length)
            datasets['train'] = train_dataset
            print("훈련 데이터 전처리 완료.")

        if self.val_path:
            print("검증 데이터 로드 중...")
            val_dataset = DatasetManager(self.val_path, self.tokenizer, self.max_length)
            datasets['val'] = val_dataset
            print("검증 데이터 전처리 완료.")

        print("데이터셋 로드 및 전처리 완료.")
        return datasets
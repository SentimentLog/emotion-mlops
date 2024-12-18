from transformers import BertForSequenceClassification, AutoTokenizer
from dotenv import load_dotenv
from huggingface_hub import login
import torch

from .configuration import Configuration
from .tokenization_kobert import KoBertTokenizer

import os

class ModelManager:
    def __init__(self, base_model="monologg/kobert", device="cpu"):
        """
        모델 초기화
        :param base_model: 베이스 모델
        :param device: 사용할 디바이스 (cuda)
        """

        self.base_model = base_model
        self.token = self._load_environment()
        self.device = device

    @staticmethod
    def _load_environment():
        current_dir = os.path.dirname(os.path.abspath(__file__))
        dotenv_path = os.path.join(current_dir, '.env')
        load_dotenv(dotenv_path)

        token = os.getenv("HUGGINGFACE_TOKEN")
        if not token:
            raise ValueError("Hugging Face 토큰을 .env 파일에 설정해주세요. 예: HUGGINGFACE_TOKEN=your_token")
        return token

    def _login_huggingface(self):
        login(token=self.token)
        print("허깅페이스 로그인 완료")

    def load_models(self, num_labels):
        """
        모델, 토크나이저, 그리고 분류를 위한 레이블 개수 확인
        :param num_labels: 레이블 개수
        :return:
        """
        config = Configuration().set_config(self.base_model)
        model = BertForSequenceClassification.from_pretrained(self.base_model, num_labels=num_labels, config=config, trust_remote_code=True)
        model.to(self.device)

        print(f"{self.base_model} 모델과 토크나이저가 성공적으로 로드 되었음")
        return model

    def load_tokenizer(self):
        tokenizer = KoBertTokenizer.from_pretrained(self.base_model, trust_remote_code=True)
        return tokenizer

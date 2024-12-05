from transformers import AutoModelForSequenceClassification, AutoTokenizer
from dotenv import load_dotenv
from huggingface_hub import login
import torch
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

    def _load_model_and_tokenizer(self, num_labels):
        """
        모델, 토크나이저, 그리고 분류를 위한 레이블 개수 확인
        :param num_labels: 레이블 개수
        :return:
        """
        model = AutoModelForSequenceClassification.from_pretrained(self.base_model, num_labels=num_labels, use_auth_token=self.token)
        tokenizer = AutoTokenizer.from_pretrained(self.base_model, trust_remote_code=True, token=self.token)

        # padding
        tokenizer.padding_side = "right"  # 패딩을 오른쪽에 적용
        if tokenizer.pad_token is None:
            # PAD 토큰이 정의되어 있지 않은 경우, EOS 토큰을 PAD 토큰으로 설정
            tokenizer.pad_token = tokenizer.eos_token
            print(f"PAD 토큰이 정의되어 있지 않아 EOS 토큰을 PAD 토큰으로 설정했습니다: {tokenizer.pad_token}")

        model.to(self.device)

        print(f"{self.base_model} 모델과 토크나이저가 성공적으로 로드 되었음")
        return model, tokenizer
from transformers import AutoModelForSequenceClassification, BertTokenizer
import torch


class InferenceHandler:
    def __init__(self, model_path, tokenizer_path, device=None):
        # 기본적으로 CPU 또는 사용 가능한 GPU 선택
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

        # 모델과 토크나이저 로드
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        self.model.to(self.device)  # 선택한 장치로 모델 이동

    def predict(self, text, emotion_labels=None):
        # Tokenization
        tokens = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)

        # 입력 데이터를 CPU 또는 GPU로 이동
        tokens = {key: val.to(self.device) for key, val in tokens.items()}

        # 모델 추론
        with torch.no_grad():
            outputs = self.model(**tokens)

        # 소프트맥스 적용하여 확률 계산
        probabilities = torch.softmax(outputs.logits, dim=-1).squeeze(0).tolist()
        predicted_label_idx = torch.argmax(outputs.logits, dim=-1).item()

        # 레이블 매핑
        if emotion_labels:
            predicted_label = emotion_labels[predicted_label_idx]
            emotion_probs = {label: prob for label, prob in zip(emotion_labels, probabilities)}
        else:
            predicted_label = predicted_label_idx
            emotion_probs = {str(i): prob for i, prob in enumerate(probabilities)}

        return {
            "text": text,
            "predicted_label": predicted_label,
            "confidence": probabilities[predicted_label_idx],
            "emotion_probabilities": emotion_probs,
        }
from transformers import BertConfig
class Configuration:
    def __init__(self):
        self.id2label = {
            0: "분노",
            1: "혐오",
            2: "중립",
            3: "놀람",
            4: "행복",
            5: "공포",
            6: "슬픔"
        }
        self.label2id = {v: k for k, v in self.id2label.items()}


        # Config 업데이트
    def set_config(self, base_model):
        config = BertConfig.from_pretrained(
            base_model,
            num_labels=7,
            id2label=self.id2label,
            label2id=self.label2id,
            author='SEXY CHEOL(choerish.hw@gmail.com)'
        )

        return config
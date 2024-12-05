from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
import os
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Tracking Module
from .tracking_manager import TrackingModel

class TraningManager:

    @staticmethod
    def compute_metrics(pred):
        """
        모델 검증용 함수
        :param pred: 모델
        :return: 평가지표
        """

        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)

        accuracy = accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
        }

    @staticmethod
    def configure_trainning(output_dir, num_train_epochs=5, learning_rate=2e-5, batch_size=32, report_to="wandb"):
        return TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_metric="accuracy",
            save_total_limit=3,
            fp16=True,
            report_to=report_to,
        )

    @staticmethod
    def train_model(model, train_dataset, eval_dataset, trainning_args, early_stopping_patience=2):
        try:
            trainer = Trainer(
                model=model,
                args=trainning_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                compute_metrics=TranningManager.compute_metrics,
                callbacks=[EarlyStoppingCallback(early_stopping_patience)],
            )

            # 학습 시작
            print(f'학습시작: 경로 -> {trainning_args.output_dir}')
            trainer.train()

            if eval_dataset:
                metrics = trainer.evaluate()
                print(f'검증 결과: {metrics}')
                TrackingModel.log_metrics(trainning_args.output_dir)


            #모델 저장
            save_path = os.path.join(training_args.output_dir, 'KoBERT-Sentiment-Analysis')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            trainer.save_model(save_path)
            print(f"{save_path} 경로로 모델 저장 완료")

            # 세션 종료
            TrackingModel.finish_wandb()

        except Exception as e:
            print(f'훈련 중 오류 발생 : {e}')
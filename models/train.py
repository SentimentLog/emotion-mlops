import argparse
import torch
import os

from datetime import datetime

from src.model_manager import ModelManager
from src.traning_manager import TraningManager
from src.training_log import TrainingLog

from data.dataset_manager import DatasetManager


def parser_args():
    parser = argparse.ArgumentParser(descri아ption='모델 파라미터 튜닝')

    parser.add_argument(
        '--train_path', type=str, default='./data/raw/train_datasets.json',
        help='훈련 데이터셋 로드, json으로 구성'
    )

    parser.add_argument(
        '--val_path', type=str, default='./data/raw/test_datasets.json',
        help='검증 데이터셋 로드, json으로 구성'
    )

    parser.add_argument(
        '--model_save_path', type=str, default='models/result',
        help='모델 저장소'
    )

    parser.add_argument(
        '--max_length', type=int, default=128,
        help='시퀀스 최대 길이'
    )

    parser.add_argument(
        '--batch_size', type=int, default=64,
        help='훈련용 배치사이즈, 작은 모델이니만큼 배치사이즈 크게 잡아도 무방'
    )
    parser.add_argument(
        '--learning_rate', type=float, default=2e-5,
        help='학습률 조정'
    )
    parser.add_argument(
        '--num_epochs', type=int, default=10,
        help='기본 5로 하긴 했는데 10으로 조정'
    )

    parser.add_argument(
        '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
        help='혹시 몰라서 GPU 세팅, T4로 해도 30분안에 학습 끝남'
    )

    parser.add_argument("--run_name", type=str, default=None,
        help="WandB 실행 이름"
    )

    parser.add_argument(
        '--project_name', type=str, required=True,
        help='WandB 프로젝트 이름'
    )


    args = parser.parse_args()
    return args

def main():
    args = parser_args()

    # CUDA 설정 확인
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("GPU 사용 불가 CPU로 실행됩니다")
        args.device = 'cpu'
    else:
        print(f'사용할 디바이스: {args.device}')


    model_manager = ModelManager(base_model='monologg/kobert', device=args.device)
    model = model_manager.load_models(num_labels=7)
    tokenizer = model_manager.load_tokenizer()

    manager = DatasetManager()
    train = manager.load_data(args.train_path).map(manager.tokenizer_settings, batched=True)
    valid = manager.load_data(args.val_path).map(manager.tokenizer_settings, batched=True)

    traning_manager = TraningManager(project_name=args.project_name, run_name=args.run_name)

    training_args = TraningManager.configure_trainning(
        output_dir=args.model_save_path,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
    )

    # 모델 학습
    try:
        print("모델 학습 시작")
        result = traning_manager.train_model(model, tokenizer, train, valid, training_args)
        print(f"학습 완료. 모델 저장 경로: {args.model_save_path}")

        training_log = TrainingLog(
            host=os.getenv("POSTGRES_HOST"),
            port=os.getenv("POSTGRES_PORT"),
            user=os.getenv("POSTGRES_USER"),
            password=os.getenv("POSTGRES_PASSWORD"),
            database=os.getenv("POSTGRES_DB"),
        )

        # 훈련 결과 저장
        training_log.save_training_log(
            model_name="KoBERT_setiment_model",
            model_version="v1.0",
            dataset_version="v1.0",
            accuracy=result['accuracy'],
            loss=result['loss'],
            f1_score=result['f1_score'],
            hyperparameters={
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "num_epochs": args.num_epochs,
                "max_length": args.max_length,
            },
            artifact_path=args.model_save_path,
            measurement_date=datetime.now().strftime("%Y%m%d%H%M%S"),
        )
    except Exception as e:
        print(f"모델 학습 중 오류 발생: {e}")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"프로그램 실행 중 오류 발생: {e}")
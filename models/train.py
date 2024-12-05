import argparse
import torch

from src.model_manager import ModelManager
from src.traning_manager import TraningManager

from data.dataset_loader import DatasetLoader


def parser_args():
    parser = argparse.ArgumentParser(description='모델 파라미터 튜닝')

    parser.add_argument(
        '--train_path', type=str, default='../data/raw/train_dataset.json',
        help='훈련 데이터셋 로드, json으로 구성'
    )

    parser.add_argument(
        '--val_path', type=str, default='../data/raw/test_dataset.json',
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
        '--batch_size', type=int, default=128,
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

    args = parser.parse_args()
    return args

def main():
    args = parser_args()
    num_labels = 7 # 추후에 전용 클래스 만들 예정

    # CUDA 설정 확인
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("GPU 사용 불가 CPU로 실행됩니다")
        args.device = 'cpu'
    else:
        print(f'사용할 디바이스: {args.device}')


    model_manager = ModelManager(base_model='monologg/kobert', device=args.device)
    model, tokenizer = model_manager._load_model_and_tokenizer(num_labels=num_labels)


    data_loader = DatasetLoader(
        train_path=args.train_path,
        val_path=args.val_path,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )

    datasets = data_loader.get_datasets()
    train = datasets['train']
    valid = datasets['valid']


    training_args = TraningManager.configure_trainning(
        output_dir=args.model_save_path,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
    )

    TraningManager.train_model(model, train, valid, training_args)


if __name__ == '__main__':
    main()
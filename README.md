# emotion-mlops

- 기획의도 
  - KoBERT를 HuggingFace에서 업로드 후 주기적인 학습을 통한 실제 사이트 운영 매커니즘 작성 

- 해당 리포지토리 역할 
  - 감정분석 모델(KoBERT) 학습 및 추론 API제공
  - 주기적인 모델 학습 및 서빙 
  - 데이터 전처리 및 학습 파이프라인 관리


- 사용 라이브러리 및 스택 
  - `Torch`, `airflow(Scheduler)`, `prefect(경량화된 Scheduler)`, `HuggingFace`

- 예정 구성 디렉토리
```bash
emotion-mlops/
├── app/
│   ├── data/
│   │   ├── preprocess.py          # 데이터 전처리 코드
│   │   ├── dataset_loader.py      # 데이터 로드 코드
│   │   └── embeddings/            # 텍스트 임베딩 데이터
│   ├── models/
│   │   ├── train.py               # KoBERT 학습 코드
│   │   ├── evaluate.py            # 모델 평가 코드
│   │   ├── checkpoints/           # 학습 중간 체크포인트
│   │   └── trained/               # 최종 학습된 모델
│   ├── serving/
│   │   ├── inference.py           # 추론 코드
│   │   ├── api_server.py          # FastAPI 기반 서빙 API
│   │   └── utils.py               # 유틸리티 코드
│   └── scheduler/
│       ├── training_flow.py       # 학습 워크플로우
│       └── schedule_tasks.py      # Prefect 스케줄링 코드
├── Dockerfile                     # Docker 설정 파일
├── requirements.txt               # Python 의존성
└── README.md                      # 프로젝트 설명
```

> 상기 디렉토리 flow는 상황에 따라 변동 될 수 있음
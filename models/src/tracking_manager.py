import wandb
import os
from dotenv import load_dotenv

class TrackingModel:
    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        dotenv_path = os.path.join(current_dir, '.env')
        load_dotenv(dotenv_path)
    @staticmethod
    def intialize(project_name, run_name=None, config=None, entity="icucheol"):
        wandb.login(key=os.getenv("WANDB_PASSWORD"))
        id = wandb.util.generate_id() # identity generate
        wandb.init(
            project=project_name,
            id=id,
            name=run_name,
            entity=entity,
            config=config,
        )
        print(f"WandB 프로젝트 '{project_name}' 초기화 완료 실행 ID: {id}")

    @staticmethod
    def log_model_path(save_path):
        """
        WandB에 모델 저장 경로 기록 및 아티팩트 업로드

        :param save_path: 모델 저장 디렉토리
        """
        if not os.path.exists(save_path):
            raise FileNotFoundError(f"경로 '{save_path}'가 존재하지 않습니다.")

        # 모델 경로 로깅
        wandb.log({"model_save_path": save_path})
        print(f"모델 저장 경로 '{save_path}'를 WandB에 기록했습니다.")

        # 모델 파일 업로드
        artifact = wandb.Artifact("trained_model", type="model")
        artifact.add_dir(save_path)
        wandb.log_artifact(artifact)
        print(f"모델 '{save_path}'를 WandB에 아티팩트로 업로드 완료.")

    @staticmethod
    def finish_wandb():
        wandb.finish()
        print("WandB 세션 종료")


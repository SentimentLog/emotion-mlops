from google.cloud import storage
import os
from datetime import datetime
from dotenv import load_dotenv

class ArtifactUploader:
    def __init__(self, file_path='../models/result/KoBERT-Sentiment-Analysis'):
        self.file_path = file_path

        current_dir = os.path.dirname(os.path.abspath(__file__))
        dotenv_path = os.path.join(current_dir, '.env')
        load_dotenv(dotenv_path)

        self.bucket_name = os.getenv('BUCKET_NAME')
        self.bucket_blob = os.getenv('BUCKET_BLOB')

        if not self.bucket_name or not self.bucket_blob:
            raise ValueError("버킷이름, 버킷경로 제대로 설정했는지 확인 → .env")

    def _get_gcs_client(self):
        try:
            return storage.Client()
        except Exception as e:
            raise RuntimeError(f"로그인 실패 : {e}")

    def upload(self):
        try:
            client = self._get_gcs_client()
            bucket = client.bucket(self.bucket_name)

            if not os.path.exists(self.file_path):
                raise FileNotFoundError(f"파일 못찾음 : {self.file_path}")

            # Generate a unique blob name with timestamp
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            destination_blob_name = f"{self.bucket_blob}_{timestamp}"

            blob = bucket.blob(destination_blob_name)
            blob.upload_from_filename(self.file_path)

            print(f"모델 업로드 완료 링크 gc://{self.bucket_name}/{destination_blob_name}")
        except FileNotFoundError as fnf_error:
            print(f"File error: {fnf_error}")
        except Exception as e:
            print(f"업로드 실패: {e}")

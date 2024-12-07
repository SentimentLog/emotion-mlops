import psycopg2
import json
from datetime import datetime

class TrainingLog:
    def __init__(self, host, port, database, user, password):
        self.conn = psycopg2.connect(
            host=host,
            database=database,
            user=user,
            password=password,
            port=port
        )

    def save_training_log(
        self,
        model_name,
        model_version,
        dataset_version,
        accuracy,
        loss,
        f1_score,
        hyperparameters,
        artifact_path,
        measurement_date=None
    ):
        try:
            if not isinstance(hyperparameters, dict):
                raise ValueError("하이퍼 파라미터는 반드시 딕셔너리여야 함!")

            if measurement_date is None:
                measurement_date = datetime.now()

            cursor = self.conn.cursor()
            query = """
                INSERT INTO model_training_logs (
                    model_name, model_version, dataset_version, 
                    accuracy, loss, f1_score, hyperparameters, 
                    artifact_path, measurement_date
                ) 
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            cursor.execute(
                query, (
                    model_name,
                    model_version,
                    dataset_version,
                    accuracy,
                    loss,
                    f1_score,
                    json.dumps(hyperparameters),
                    artifact_path,
                    measurement_date
                )
            )
            self.conn.commit()
            print("훈련 내용 저장 완료")

        except Exception as e:
            print(f"훈련 저장 문제 발생 : {e}")

        finally:
            cursor.close()

    def close_connection(self):
        self.conn.close()
from .src.inference import InferenceHandler
from http.client import HTTPException

from fastapi import APIRouter, HTTPException
from .query_schemas import ModelQuery, ReturnQuery

EMOTION_LABELS = ["분노", "혐오", "중립", "놀람", "행복", "공포", "슬픔"]
model_path = "UICHEOL-HWANG/kobert"
tokenizer_path = "UICHEOL-HWANG/kobert"
Bert = InferenceHandler(model_path, tokenizer_path)


query_router = APIRouter(
    prefix="/query",
    tags=["query"],
    responses={404: {"description": "Not found"}},
)

@query_router.post("/", response_model=ReturnQuery)
async def model_query(query: ModelQuery):

    try:
        answer_hash = Bert.predict(query.query, EMOTION_LABELS)
        return ReturnQuery(answer_hash=answer_hash)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


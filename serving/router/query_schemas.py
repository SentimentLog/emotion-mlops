from typing import Dict
from pydantic import BaseModel

class ModelQuery(BaseModel):
    query : str

class ReturnQuery(BaseModel):
    return_query : Dict[str, str]

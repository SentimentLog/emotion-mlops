import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from router.query import query_router

app = FastAPI(root_path='/model_api/')
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(query_router)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
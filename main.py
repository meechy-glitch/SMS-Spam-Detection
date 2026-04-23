from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# from database.database import engine
# from database.models import Base
from routes import pages, predict

# Base.metadata.create_all(bind=engine)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(pages.router)
app.include_router(predict.router)

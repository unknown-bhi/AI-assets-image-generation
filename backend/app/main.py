

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import settings
from app.modules.auth.router import router as auth_router
from app.modules.generation.router import router as generation_router


@asynccontextmanager
async def lifespan(app: FastAPI):
  
    print(f"Starting {settings.APP_NAME} in {settings.ENVIRONMENT} mode")
    # Storage dirs ensure karo
    settings.ensure_dirs()
    yield
    print("Shutting down...")


app = FastAPI(
    title=settings.APP_NAME,
    version="0.1.0",
    debug=settings.DEBUG,
    lifespan=lifespan,
)


app.include_router(auth_router ,prefix="/api/v1")
app.include_router(generation_router, prefix= '/api/v1')

@app.get("/health", tags=["health"])
async def health():
    return {"status": "ok", "app": settings.APP_NAME}
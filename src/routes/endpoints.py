from fastapi import APIRouter

router = APIRouter()


@router.get("/ping")
def ping():
    return {"message": "pong"}


def setup_routes(app):
    app.include_router(router)

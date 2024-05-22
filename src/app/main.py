# from fastapi import Depends, FastAPI, HTTPException, Response, status
# from sqlalchemy.orm import Session

# from app.db.database import Base, engine, get_db
# from app.db.models import Detection, Image
# from app.db.repositories import DetectionRepository, ImageRepository
# from app.schemas.image import DetectionRequest, DetectionResponse, ImageRequest, ImageResponse

# Base.metadata.create_all(bind=engine)

# app = FastAPI()


# @app.post("/api/images", response_model=ImageResponse, status_code=status.HTTP_201_CREATED)
# def create(request: ImageRequest, db: Session = Depends(get_db)):
#     image = ImageRepository.save(db, ImageRepository(**request.dict()))
#     return ImageResponse.from_orm(image)

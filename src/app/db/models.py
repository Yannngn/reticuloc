from sqlalchemy import ARRAY, Column, Float, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from app.db.database import Base


class Image(Base):
    __tablename__ = "images"

    id = Column(Integer, primary_key=True, index=True)
    image = Column(String)
    detections = relationship("Detection", back_populates="image")


class Detection(Base):
    __tablename__ = "detections"

    id = Column(Integer, primary_key=True, index=True)
    image_id = Column(Integer, ForeignKey("images.id"))
    label = Column(String)
    conf = Column(Float)
    box = Column(ARRAY(Float))
    image = relationship("Image", back_populates="detections")

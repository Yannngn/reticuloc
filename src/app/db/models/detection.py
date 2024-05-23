import os
import sys

from sqlalchemy import ARRAY, Column, Float, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))

from src.app.db.database import Base


class DbDetection(Base):
    __tablename__ = "detection"
    id = Column(Integer, primary_key=True, index=True)
    label = Column(String)
    confidence = Column(Float)
    box = Column(ARRAY(Integer))  # Talvez o melhor seja xyxyn

    # só image id ou adicionar user id ?
    image_id = Column(Integer, ForeignKey("image.id"))
    user_id = Column(Integer, ForeignKey("user.id"))

    image = relationship("DbImage")
    user = relationship("DbUser")


class DbDetectionCrop(Base):
    __tablename__ = "detection_crop"
    id = Column(Integer, primary_key=True, index=True)
    data = Column(String)  # recorte da imagem em base64
    label = Column(String)
    confidence = Column(Float)
    box = Column(ARRAY(Integer))  # Talvez o melhor seja xyxyn

    # só image id ou adicionar user id ?
    image_id = Column(Integer, ForeignKey("image.id"))
    user_id = Column(Integer, ForeignKey("user.id"))

    image = relationship("DbImage")
    user = relationship("DbUser")


class DbDetectionCount(Base):
    __tablename__ = "detection_count"
    id = Column(Integer, primary_key=True, index=True)
    erythrocyte_count = Column(Integer)
    punctated_reticulocyte_count = Column(Integer)
    aggregate_reticulocyte_count = Column(Integer)
    erythrocyte_conf = Column(Float)
    punctated_reticulocyte_conf = Column(Float)
    aggregate_reticulocyte_conf = Column(Float)

    # só image id ou adicionar user id ?
    image_id = Column(Integer, ForeignKey("image.id"))
    user_id = Column(Integer, ForeignKey("user.id"))

    image = relationship("DbImage")
    user = relationship("DbUser")

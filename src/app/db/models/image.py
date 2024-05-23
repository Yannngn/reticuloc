import os
import sys

from sqlalchemy import Column, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))

from src.app.db.database import Base


class DbImage(Base):
    __tablename__ = "image"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("user.id"))
    image_data = Column(String)

    user = relationship("DbUser", back_populates="images")

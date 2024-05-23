import os
import sys

from sqlalchemy import Column, Integer, String

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))

from src.app.db.database import Base


class DbUser(Base):
    __tablename__ = "user"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String)
    email = Column(String)
    password = Column(String)

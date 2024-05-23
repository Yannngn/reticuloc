from db.database import get_db
from db.models.user import DbUser
from fastapi import Depends, FastAPI
from schemas.user import UserDisplay
from sqlalchemy.orm.session import Session

app = FastAPI()


@app.get("/all", response_model=list[UserDisplay])
def get_users(db: Session = Depends(get_db)):
    return db.query(DbUser).all()

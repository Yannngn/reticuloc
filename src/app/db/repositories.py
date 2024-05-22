from sqlalchemy.orm import Session

from app.db.models import Detection, Image


class DetectionRepository:
    @staticmethod
    def find_all(db: Session) -> list[Detection]:
        return db.query(Detection).all()

    @staticmethod
    def save(db: Session, detection: Detection) -> Detection:
        if detection.id is not None:
            db.merge(detection)
        else:
            db.add(detection)
        db.commit()
        return detection

    @staticmethod
    def find_by_id(db: Session, id: int) -> Detection:
        return db.query(Detection).filter(Detection.id == id).first()

    @staticmethod
    def find_by_image_id(db: Session, id: int) -> Detection:
        return db.query(Detection).filter(Detection.image_id == id).first()

    @staticmethod
    def exists_by_id(db: Session, id: int) -> bool:
        return db.query(Detection).filter(Detection.id == id).first() is not None

    @staticmethod
    def exists_by_image_id(db: Session, id: int) -> bool:
        return db.query(Detection).filter(Detection.image_id == id).first() is not None

    @staticmethod
    def delete_by_id(db: Session, id: int) -> None:
        detection = db.query(Detection).filter(Detection.id == id).first()
        if detection is not None:
            db.delete(detection)
            db.commit()


class ImageRepository:
    @staticmethod
    def find_all(db: Session) -> list[Image]:
        return db.query(Image).all()

    @staticmethod
    def save(db: Session, image: Image) -> Image:
        if image.id is not None:
            db.merge(image)
        else:
            db.add(image)
        db.commit()
        return image

    @staticmethod
    def find_by_id(db: Session, id: int) -> Image:
        return db.query(Image).filter(Image.id == id).first()

    @staticmethod
    def exists_by_id(db: Session, id: int) -> bool:
        return db.query(Image).filter(Image.id == id).first() is not None

    @staticmethod
    def delete_by_id(db: Session, id: int) -> None:
        detection = db.query(Image).filter(Image.id == id).first()
        if detection is not None:
            db.delete(detection)
            db.commit()

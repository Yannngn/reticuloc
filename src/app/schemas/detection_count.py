from pydantic import BaseModel


class DetectionCountBase(BaseModel):
    aggregate_reticulocyte_count: int
    aggregate_reticulocyte_conf: float
    punctate_reticulocyte_count: int
    punctate_reticulocyte_conf: float
    erythrocyte_count: int
    erythrocyte_conf: float

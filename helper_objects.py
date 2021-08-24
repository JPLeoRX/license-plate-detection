from typing import List
from PIL.Image import Image
from pydantic import BaseModel
from simplestr import gen_str_repr_eq


@gen_str_repr_eq
class Rectangle(BaseModel):
    x: float
    y: float
    w: float
    h: float

    def __init__(self, x: float, y: float, w: float, h: float) -> None:
        super().__init__(x=x, y=y, w=w, h=h)


@gen_str_repr_eq
class LabeledBox(BaseModel):
    label: str
    region: Rectangle # Relative coordinates, from 0.0 to 1.0

    def __init__(self, label: str, region: Rectangle) -> None:
        super().__init__(label=label, region=region)


@gen_str_repr_eq
class Sample(BaseModel):
    name: str
    image: Image
    boxes: List[LabeledBox]

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, name: str, image: Image, boxes: List[LabeledBox]) -> None:
        super().__init__(name=name, image=image, boxes=boxes)


@gen_str_repr_eq
class Prediction(BaseModel):
    label: str
    score: float
    region: Rectangle

    def __init__(self, label: str, score: float, region: Rectangle) -> None:
        super().__init__(label=label, score=score, region=region)

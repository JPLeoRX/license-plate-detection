import tempfile
import uuid
import cv2
import numpy
from PIL.Image import Image
from PIL import Image as image_main


def generate_uuid() -> str:
    return str(uuid.uuid4())


def open_image_pil(image_path: str) -> Image:
    return image_main.open(image_path)


def save_image_pil(pil_image: Image, image_path: str) -> str:
    pil_image.save(image_path)
    return image_path


def save_image_pil_in_temp(pil_image: Image, image_name: str) -> str:
    folder_path = str(tempfile.gettempdir())
    image_path = folder_path + '/' + image_name + '.' + pil_image.format.lower()
    return save_image_pil(pil_image, image_path)


def convert_pil_to_cv(pil_image: Image):
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')
    return cv2.cvtColor(numpy.array(pil_image), cv2.COLOR_RGB2BGR)


def debug_image_cv(cv_image):
    cv2.namedWindow('Debug Image', cv2.WINDOW_NORMAL)
    cv2.imshow('Debug Image', cv_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def debug_image_pil(pil_image: Image):
    debug_image_cv(convert_pil_to_cv(pil_image))

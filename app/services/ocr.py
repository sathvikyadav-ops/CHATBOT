# ==========================================================
# IMPORTS
# ==========================================================
import io

import pytesseract
from PIL import (
    Image,
    ImageOps
)

from app.config import Config


# ==========================================================
# OCR ENGINE
# ==========================================================
class OCREngine:

    def __init__(self):

        pytesseract.pytesseract.tesseract_cmd = (
            Config.TESSERACT_PATH
        )

        self.dpi = Config.OCR_DPI

    # ======================================================
    # PREPROCESS IMAGE
    # ======================================================
    def _preprocess(
        self,
        image: Image.Image
    ) -> Image.Image:

        image = image.convert("L")

        image = ImageOps.autocontrast(
            image
        )

        return image

    # ======================================================
    # OCR PDF PAGE
    # ======================================================
    def run_ocr(
        self,
        page
    ) -> str:

        try:

            pix = page.get_pixmap(
                dpi=self.dpi
            )

            image = Image.open(
                io.BytesIO(
                    pix.tobytes("png")
                )
            ).convert("RGB")

            image = self._preprocess(
                image
            )

            text = (
                pytesseract.image_to_string(
                    image,
                    lang="eng",
                    config="--oem 3 --psm 3"
                )
            )

            return text.strip()

        except Exception as e:

            print(
                f"[OCR ERROR] {e}"
            )

            return ""

    # ======================================================
    # OCR IMAGE FILE
    # ======================================================
    def run_image_ocr(
        self,
        image_path: str
    ) -> str:

        try:

            image = Image.open(
                image_path
            ).convert("RGB")

            image = self._preprocess(
                image
            )

            text = (
                pytesseract.image_to_string(
                    image,
                    lang="eng",
                    config="--oem 3 --psm 3"
                )
            )

            return text.strip()

        except Exception as e:

            print(
                f"[IMAGE OCR ERROR] {e}"
            )

            return ""
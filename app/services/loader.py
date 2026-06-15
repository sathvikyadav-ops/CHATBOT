# ==========================================================
# IMPORTS
# ==========================================================
import fitz as pymupdf

from app.services.ocr import OCREngine
from app.utils.text_utils import TextCleaner


# ==========================================================
# PDF LOADER
# ==========================================================
class PDFLoader:

    def __init__(self):

        self.ocr = OCREngine()

        self.cleaner = TextCleaner()

    # ======================================================
    # EXTRACT PAGE TEXT
    # ======================================================
    def extract_page_text(
        self,
        page
    ) -> str:

        text = page.get_text(
            "text"
        )

        ocr_text = (
            self.ocr.run_ocr(
                page
            )
        )

        combined = "\n".join(
            filter(
                None,
                [
                    text,
                    ocr_text
                ]
            )
        )

        return (
            self.cleaner.clean_text(
                combined
            )
        )

    # ======================================================
    # EXTRACT PDF PAGE-WISE
    # ======================================================
    def extract_text_from_pdf(
        self,
        file_path: str
    ) -> list:

        try:

            doc = pymupdf.open(
                file_path
            )

            pages = []

            for page_number, page in enumerate(
                doc,
                start=1
            ):

                page_text = (
                    self.extract_page_text(
                        page
                    )
                )

                if page_text.strip():

                    pages.append(
                        {
                            "page_number":
                            page_number,

                            "text":
                            page_text
                        }
                    )

            doc.close()

            return pages

        except Exception as e:

            print(
                f"[PDF ERROR] {e}"
            )

            return []
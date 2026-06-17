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

        try:

            # ----------------------------------------------
            # TRY NORMAL PDF TEXT EXTRACTION FIRST
            # ----------------------------------------------
            text = page.get_text(
                "text"
            ).strip()

            # ----------------------------------------------
            # IF TEXT EXISTS, SKIP OCR
            # ----------------------------------------------
            if text:

                return (
                    self.cleaner.clean_text(
                        text
                    )
                )

            # ----------------------------------------------
            # OTHERWISE RUN OCR
            # ----------------------------------------------
            print(
                f"[OCR] Page "
                f"{page.number + 1}"
            )

            ocr_text = (
                self.ocr.run_ocr(
                    page
                )
            )

            return (
                self.cleaner.clean_text(
                    ocr_text
                )
            )

        except Exception as e:

            print(
                f"[PAGE EXTRACTION ERROR] {e}"
            )

            return ""

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

            total_pages = len(doc)

            print(
                f"Processing "
                f"{total_pages} pages..."
            )

            for (
                page_number,
                page
            ) in enumerate(
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

            print(
                f"Extracted "
                f"{len(pages)} pages"
            )

            return pages

        except Exception as e:

            print(
                f"[PDF ERROR] {e}"
            )

            return []
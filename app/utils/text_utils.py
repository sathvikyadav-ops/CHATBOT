# =========================
# IMPORTS
# =========================
import re


# =========================
# TEXT CLEANER CLASS
# =========================
class TextCleaner:

    # =========================
    # NORMALIZE TEXT
    # =========================
    def normalize_text(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    # =========================
    # REMOVE EXTRA NEWLINES
    # =========================
    def remove_extra_newlines(self, text: str) -> str:
        text = re.sub(r"\n+", "\n", text)
        return text.strip()

    # =========================
    # REMOVE SPECIAL CHARACTERS
    # =========================
    def remove_special_characters(self, text: str) -> str:
        text = re.sub(r"[^\w\s.,!?;:()/-]", "", text)
        return text

    # =========================
    # MAIN CLEANING PIPELINE
    # =========================
    def clean_text(self, text: str) -> str:
        text = self.normalize_text(text)
        text = self.remove_extra_newlines(text)
        text = self.remove_special_characters(text)
        return text
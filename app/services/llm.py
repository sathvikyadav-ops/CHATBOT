# =========================
# IMPORTS
# =========================
import requests

from app.config import Config
from app.prompt import RAG_SYSTEM_PROMPT


# =========================
# GROQ LLM CLASS
# =========================
class GroqLLM:

    def __init__(self):

        self.api_key = Config.GROQ_API_KEY
        self.model = Config.GROQ_MODEL
        self.url = Config.GROQ_API_URL

        if not self.api_key:
            raise ValueError("GROQ_API_KEY is missing in .env")

    # =========================
    # GENERATE RESPONSE
    # =========================
    def generate(self, query, context):

        prompt = f"""
Context:

{context}

Question:

{query}
"""

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": RAG_SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.2
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        response = requests.post(
            self.url,
            headers=headers,
            json=payload,
            timeout=120
        )

        # =========================
        # ERROR HANDLING
        # =========================
        if response.status_code != 200:
            raise Exception(
                f"Groq/XAI API Error {response.status_code}: {response.text}"
            )

        data = response.json()

        return data["choices"][0]["message"]["content"]
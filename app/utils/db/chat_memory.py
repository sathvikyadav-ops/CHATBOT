# ==========================================================
# IMPORTS
# ==========================================================
from pymongo import MongoClient
from app.config import Config


# ==========================================================
# CHAT MEMORY
# ==========================================================
class ChatMemory:

    def __init__(self):

        client = MongoClient(
            Config.MONGO_URI
        )

        db = client[
            Config.MONGO_DB
        ]

        self.collection = db[
            Config.CHAT_COLLECTION
        ]

    # ======================================================
    # SAVE CHAT
    # ======================================================
    def save_message(
        self,
        session_id: str,
        query: str,
        answer: str
    ):

        self.collection.insert_one(
            {
                "session_id": session_id,
                "query": query,
                "answer": answer
            }
        )

    # ======================================================
    # GET LAST N MESSAGES
    # ======================================================
    def get_history(
        self,
        session_id: str
    ):

        history = list(
            self.collection
            .find(
                {"session_id": session_id},
                {"_id": 0}
            )
            .sort("_id", -1)
            .limit(
                Config.CHAT_HISTORY_LIMIT
            )
        )

        history.reverse()

        return history
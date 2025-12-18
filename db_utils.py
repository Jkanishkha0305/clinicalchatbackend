import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

# Environment variable names we accept for Atlas
ATLAS_URI_ENV_KEYS = ("MONGODB_URI", "MONGO_URL", "MONGODB_PROD_URL")
LOCAL_MARKERS = ("localhost", "127.0.0.1", "0.0.0.0")


def get_mongo_uri() -> str:
    """
    Return the MongoDB Atlas connection string from the environment.
    Refuses to use localhost/127.0.0.1 to avoid accidental local connections.
    """
    for key in ATLAS_URI_ENV_KEYS:
        uri = os.getenv(key)
        if uri:
            if any(marker in uri for marker in LOCAL_MARKERS):
                raise RuntimeError(
                    "MongoDB URI points to a local instance. "
                    "Set your Atlas URI in MONGODB_URI (or MONGO_URL/MONGODB_PROD_URL)."
                )
            return uri

    raise RuntimeError(
        "MongoDB Atlas URI missing. "
        "Set MONGODB_URI (or MONGO_URL/MONGODB_PROD_URL) to your Atlas connection string."
    )


def get_mongo_client(**kwargs) -> MongoClient:
    """
    Build a validated MongoDB client using the Atlas URI.
    """
    uri = get_mongo_uri()
    client = MongoClient(uri, **kwargs)
    # Validate connection early to surface misconfiguration
    client.admin.command("ping")
    return client

import os
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
from bson import ObjectId
from datetime import datetime
from typing import List, Dict, Optional

load_dotenv()

# MongoDB connection
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
DATABASE_NAME = os.getenv("DATABASE_NAME", "nlp_library")

# Lazy initialization for testing
_client = None
_db = None


def get_client():
    global _client
    if _client is None:
        _client = AsyncIOMotorClient(MONGODB_URI)
    return _client


def get_db():
    global _db
    if _db is None:
        _db = get_client()[DATABASE_NAME]
    return _db


# Collections
def get_documents():
    return get_db().documents


def get_document_stats():
    return get_db().document_stats


def get_users():
    return get_db().users


async def get_user_documents(user_id: str) -> List[Dict]:
    """Get all documents for a user"""
    cursor = get_documents().find({"user_id": user_id}).sort("uploaded_at", -1)
    docs_list = []
    async for doc in cursor:
        doc["_id"] = str(doc["_id"])
        # Convert stats_id to string if it exists
        if "stats_id" in doc and doc["stats_id"] is not None:
            doc["stats_id"] = str(doc["stats_id"])
        # Convert datetime to string for JSON serialization
        if "uploaded_at" in doc and doc["uploaded_at"]:
            doc["uploaded_at"] = doc["uploaded_at"].isoformat()
        docs_list.append(doc)
    return docs_list


async def create_document(user_id: str, filename: str) -> str:
    """Create a new document and return its ID"""
    item = {
        "user_id": user_id,
        "filename": filename,
        "uploaded_at": datetime.utcnow(),
        "status": "processing",
        "error": None,
    }
    result = await get_documents().insert_one(item)
    return str(result.inserted_id)


async def update_document(
    document_id: str, stats_id: str = None, status: str = "completed", error: str = None
):
    """Update document with stats reference and status"""
    update_data = {"status": status}
    if stats_id:
        update_data["stats_id"] = ObjectId(stats_id)
    if error:
        update_data["error"] = error

    await get_documents().update_one(
        {"_id": ObjectId(document_id)}, {"$set": update_data}
    )


async def save_document_stats(stats_data: Dict) -> str:
    """Save document stats and return the stats ID"""
    result = await get_document_stats().insert_one(stats_data)
    return str(result.inserted_id)


async def get_document(document_id: str, user_id: str) -> Optional[Dict]:
    """Get a specific document with its stats"""
    item = await get_documents().find_one(
        {"_id": ObjectId(document_id), "user_id": user_id}
    )

    if not item:
        return None

    # Convert ObjectId to string
    item["_id"] = str(item["_id"])
    # Convert stats_id to string if it exists
    if "stats_id" in item and item["stats_id"] is not None:
        item["stats_id"] = str(item["stats_id"])

    # Get stats if available
    if "stats_id" in item:
        stats = await get_document_stats().find_one({"_id": ObjectId(item["stats_id"])})
        if stats:
            stats["_id"] = str(stats["_id"])
            # Convert any other ObjectIds in stats to strings
            for key, value in stats.items():
                if (
                    hasattr(value, "__class__")
                    and value.__class__.__name__ == "ObjectId"
                ):
                    stats[key] = str(value)
            item["stats"] = stats

    return item


# User management
async def create_user(email: str, hashed_password: str) -> str:
    """Create a new user and return user ID"""
    user = {
        "email": email,
        "hashed_password": hashed_password,
        "created_at": datetime.utcnow(),
    }
    result = await get_users().insert_one(user)
    return str(result.inserted_id)


async def get_user_by_email(email: str) -> Optional[Dict]:
    """Get user by email"""
    user = await get_users().find_one({"email": email})
    if user:
        user["_id"] = str(user["_id"])
        # Convert datetime to string for JSON serialization
        if "created_at" in user and user["created_at"]:
            user["created_at"] = user["created_at"].isoformat()
    return user


async def get_user_by_id(user_id: str) -> Optional[Dict]:
    """Get user by ID"""
    user = await get_users().find_one({"_id": ObjectId(user_id)})
    if user:
        user["_id"] = str(user["_id"])
        # Convert datetime to string
        if "created_at" in user and user["created_at"]:
            user["created_at"] = user["created_at"].isoformat()
    return user


async def test_connection():
    """Test MongoDB connection"""
    try:
        await get_client().admin.command("ping")
        print("Connected to MongoDB successfully")
        return True
    except Exception as e:
        print(f"MongoDB connection failed: {e}")
        return False

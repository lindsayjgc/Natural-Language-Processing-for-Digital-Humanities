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

client = AsyncIOMotorClient(MONGODB_URI)
db = client[DATABASE_NAME]

# Collections
library_items = db.library_items
document_stats = db.document_stats


async def get_user_documents(user_id: str) -> List[Dict]:
    """Get all documents for a user"""
    cursor = library_items.find({"user_id": user_id}).sort("uploaded_at", -1)
    documents = []
    async for doc in cursor:
        doc["_id"] = str(doc["_id"])
        documents.append(doc)
    return documents


async def create_library_item(user_id: str, filename: str) -> str:
    """Create a new library item and return its ID"""
    item = {
        "user_id": user_id,
        "filename": filename,
        "uploaded_at": datetime.utcnow(),
        "status": "processing",
        "error": None,
    }
    result = await library_items.insert_one(item)
    return str(result.inserted_id)


async def update_library_item(
    item_id: str, stats_id: str = None, status: str = "completed", error: str = None
):
    """Update library item with stats reference and status"""
    update_data = {"status": status}
    if stats_id:
        update_data["stats_id"] = ObjectId(stats_id)
    if error:
        update_data["error"] = error

    await library_items.update_one({"_id": ObjectId(item_id)}, {"$set": update_data})


async def save_document_stats(stats_data: Dict) -> str:
    """Save document stats and return the stats ID"""
    result = await document_stats.insert_one(stats_data)
    return str(result.inserted_id)


async def get_library_item(item_id: str, user_id: str) -> Optional[Dict]:
    """Get a specific library item with its stats"""
    item = await library_items.find_one({"_id": ObjectId(item_id), "user_id": user_id})

    if not item:
        return None

    # Convert ObjectId to string
    item["_id"] = str(item["_id"])

    # Get stats if available
    if "stats_id" in item:
        stats = await document_stats.find_one({"_id": ObjectId(item["stats_id"])})
        if stats:
            stats["_id"] = str(stats["_id"])
            item["stats"] = stats

    return item


async def test_connection():
    """Test MongoDB connection"""
    try:
        await client.admin.command("ping")
        print("Connected to MongoDB successfully")
        return True
    except Exception as e:
        print(f"MongoDB connection failed: {e}")
        return False

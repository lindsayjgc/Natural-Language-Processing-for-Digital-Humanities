import asyncio
from pymongo import AsyncMongoClient
from dotenv import load_dotenv
import os
load_dotenv()
client = AsyncMongoClient(os.getenv("MONGODB_URI"), server_api=pymongo.server_api.ServerApi(
   version="1", strict=True, deprecation_errors=True))

# TODO: implement database interactions, this is just the boilerplate from mongo docs for async mongo
async def main():
    try:
        # start example code here

        # end example code here

        await client.admin.command("ping")
        print("Connected successfully")

        # other application code

        await client.close()

    except Exception as e:
        raise Exception(
            "The following error occurred: ", e)

asyncio.run(main())
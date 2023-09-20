from pymongo import MongoClient
import configparser

config = configparser.ConfigParser()
config.read("config.ini")


connection_string = config.get("MONGO", "CONNECTION_STRING")
db_name = config.get("MONGO", "DB_NAME")
openai_key = config.get("OPENAI", "OPENAI_KEY")

# Insert connection string and db name:
client = MongoClient(connection_string)
db = client[db_name]

chats_db = db["chats"]
messages_db = db["messages"]

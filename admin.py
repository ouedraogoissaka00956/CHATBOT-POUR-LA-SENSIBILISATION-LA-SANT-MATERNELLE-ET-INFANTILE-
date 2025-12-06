from pymongo import MongoClient
from werkzeug.security import generate_password_hash

mongo_client = MongoClient("mongodb://localhost:27017/")
db = mongo_client["chatbot_sante"]
users = db["users"]

admin_user = {
    "nom": "Admin",
    "prenom": "Principal",
    "username": "administrateur",
    "password": generate_password_hash("admin123"),
    "role": "admin",
    "verified": True
}

if users.find_one({"username": admin_user["username"]}):
    print("Admin already exists")
else:
    users.insert_one(admin_user)
    print("Admin created")

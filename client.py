from pymongo import MongoClient

# MongoDB local
local_client = MongoClient("mongodb://localhost:27017/")
local_db = local_client["chatbot_sante"]

# MongoDB Atlas
atlas_uri = "mongodb+srv://Issaka_sako:issaka7894561230@cluster0.fogdwsi.mongodb.net/?appName=Cluster0"
atlas_client = MongoClient(atlas_uri)
atlas_db = atlas_client["chatbot_sante"]

# Collections à migrer
collections = [
    "users", 
    "conversations", 
    "consultations", 
    "reminders",
    "alerts",
    "messages_privés",
    "weekly_tips"
]

print("🔄 Migration en cours...\n")

for coll_name in collections:
    local_coll = local_db[coll_name]
    atlas_coll = atlas_db[coll_name]
    
    docs = list(local_coll.find())
    
    if docs:
        atlas_coll.insert_many(docs)
        print(f"✅ {coll_name}: {len(docs)} documents migrés")
    else:
        print(f"⚠️  {coll_name}: Aucun document")

print("\n✅ Migration terminée !")
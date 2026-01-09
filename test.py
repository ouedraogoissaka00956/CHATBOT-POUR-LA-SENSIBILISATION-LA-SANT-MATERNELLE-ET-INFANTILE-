# ============================================================
# check_users.py - Vérifier les utilisateurs dans MongoDB
# ============================================================

from pymongo import MongoClient
from werkzeug.security import check_password_hash, generate_password_hash
import os

# Connexion MongoDB
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
client = MongoClient(MONGO_URI)
db = client["chatbot_sante"]
users_collection = db["users"]

print("="*60)
print("🔍 VÉRIFICATION DES UTILISATEURS")
print("="*60)

# Compter les utilisateurs
total_users = users_collection.count_documents({})
print(f"\n📊 Total d'utilisateurs : {total_users}")

if total_users == 0:
    print("\n⚠️ Aucun utilisateur trouvé dans la base de données !")
    print("👉 Créez un compte via /register")
else:
    print("\n📋 Liste des utilisateurs :\n")
    
    users = users_collection.find({})
    
    for i, user in enumerate(users, 1):
        print(f"{i}. Username: {user.get('username')}")
        print(f"   Nom: {user.get('prenom')} {user.get('nom')}")
        print(f"   Email: {user.get('email')}")
        print(f"   Rôle: {user.get('role')}")
        
        # Vérifier le format du hash
        pwd_hash = user.get('password', '')
        if pwd_hash.startswith('pbkdf2:'):
            print(f"   Hash: ✅ Format moderne")
        elif pwd_hash.startswith('scrypt:'):
            print(f"   Hash: ✅ Format scrypt")
        else:
            print(f"   Hash: ⚠️ Format ancien ou inconnu")
        
        print(f"   ID: {user.get('_id')}")
        print()

print("="*60)
print("🔧 OPTIONS DE CORRECTION")
print("="*60)

# Proposer de créer un utilisateur de test
choice = input("\n❓ Voulez-vous créer un utilisateur de test ? (o/n) : ").lower()

if choice == 'o':
    print("\n📝 Création d'un utilisateur de test...")
    
    # Supprimer l'ancien utilisateur test s'il existe
    users_collection.delete_one({"username": "test"})
    
    test_password = "Test1234"
    hashed = generate_password_hash(test_password)
    
    users_collection.insert_one({
        "nom": "Test",
        "prenom": "Utilisateur",
        "username": "test",
        "password": hashed,
        "role": "patiente",
        "email": "test@example.com",
        "photo": "image.png",
        "profile": {}
    })
    
    print("\n✅ Utilisateur de test créé !")
    print(f"   Username: test")
    print(f"   Password: {test_password}")
    print(f"   Role: patiente")
    print("\n👉 Essayez de vous connecter avec ces identifiants")

# Proposer de réinitialiser un mot de passe
print("\n" + "="*60)
reset_choice = input("❓ Voulez-vous réinitialiser le mot de passe d'un utilisateur ? (o/n) : ").lower()

if reset_choice == 'o':
    username = input("👤 Entrez le username : ")
    user = users_collection.find_one({"username": username})
    
    if user:
        new_password = input("🔑 Nouveau mot de passe : ")
        new_hash = generate_password_hash(new_password)
        
        users_collection.update_one(
            {"username": username},
            {"$set": {"password": new_hash}}
        )
        
        print(f"\n✅ Mot de passe mis à jour pour {username}")
        print(f"   Nouveau mot de passe : {new_password}")
    else:
        print(f"\n❌ Utilisateur '{username}' non trouvé")

print("\n" + "="*60)
print("✅ Vérification terminée")
print("="*60)
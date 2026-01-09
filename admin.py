#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ============================================================
# create_production_admin.py - Créer un admin en production
# ============================================================

from pymongo import MongoClient
from werkzeug.security import generate_password_hash
from datetime import datetime

# 🔥 REMPLACEZ PAR VOTRE URI MONGODB ATLAS
MONGO_URI = "mongodb+srv://Issaka_sako:issaka7894561230@cluster0.fogdwsi.mongodb.net/chatbot_sante?retryWrites=true&w=majority"

print("\n" + "="*70)
print(" CRÉATION D'UN ADMIN EN PRODUCTION")
print("="*70)

try:
    # Connexion à MongoDB Atlas
    client = MongoClient(MONGO_URI)
    db = client["chatbot_sante"]
    users_collection = db["users"]
    
    # Test de connexion
    client.admin.command('ping')
    print("\n Connecté à MongoDB Atlas")
    
    # Informations du compte admin
    print("\n Entrez les informations de l'admin :\n")
    
    nom = input("Nom : ").strip()
    prenom = input("Prénom : ").strip()
    username = input("Username : ").strip()
    email = input("Email : ").strip()
    password = input("Mot de passe : ").strip()
    
    # Vérifier si existe déjà
    existing = users_collection.find_one({
        "$or": [
            {"username": username},
            {"email": email}
        ]
    })
    
    if existing:
        print(f"\n⚠ Un utilisateur existe déjà !")
        choice = input("Supprimer et recréer ? (o/n) : ").lower()
        if choice == 'o':
            users_collection.delete_one({"_id": existing["_id"]})
            print(" Ancien compte supprimé")
        else:
            print(" Annulé")
            exit()
    
    # Créer le hash
    hashed = generate_password_hash(password, method='pbkdf2:sha256')
    
    # Créer l'admin
    admin = {
        "nom": nom,
        "prenom": prenom,
        "username": username,
        "password": hashed,
        "role": "admin",
        "email": email,
        "photo": "image.png",
        "profile": {},
        "created_at": datetime.now(),
        "verified": True
    }
    
    result = users_collection.insert_one(admin)
    
    print("\n" + "="*70)
    print(" ADMIN CRÉÉ EN PRODUCTION !")
    print("="*70)
    print(f"\n Identifiants :\n")
    print(f"   Username : {username}")
    print(f"   Password : {password}")
    print(f"   Role     : admin")
    print(f"   ID       : {result.inserted_id}")
    print("\n Connectez-vous sur votre app en ligne")
    print("="*70 + "\n")
    
except Exception as e:
    print(f"\n Erreur : {e}")
    print("\n Vérifiez :")
    print("  - L'URI MongoDB Atlas est correct")
    print("  - Votre IP est autorisée dans Network Access")
    print("  - L'utilisateur DB a les bonnes permissions")
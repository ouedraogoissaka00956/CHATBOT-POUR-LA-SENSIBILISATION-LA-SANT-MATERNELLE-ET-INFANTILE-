# ============================================================
# app.py — version complète avec page d’accueil + rôles + gestion utilisateurs + MongoDB
# ============================================================

from flask import Flask, render_template, request, session, redirect, url_for, flash
import numpy as np
import os
import re
import unicodedata
import pickle
import pandas as pd
from bson import ObjectId
from functools import wraps
from datetime import date

# Deep Learning
from tensorflow import keras
from tensorflow.keras import layers

# Scikit-learn
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report

# NLTK
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

# MongoDB
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash

# File upload
from werkzeug.utils import secure_filename


# Initialisation Flask
app = Flask(__name__)
# Clé de session (à changer en production)
app.secret_key = '9bcddfd4d2cb02dd798a9a990f97595cd3e3872c2865e7cf'



# ===========================================
# 📧 CONFIG EMAIL (SMTP)
# ===========================================
from flask_mail import Mail, Message

app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'oissaka439@gmail.com' 
app.config['MAIL_PASSWORD'] = 'nnaswalfkjiztdts'
app.config['MAIL_DEFAULT_SENDER'] = 'oissaka439@gmail.com'

mail = Mail(app)
# ===========================================
import re

def is_valid_email(email):
    regex = r"^[\w\.-]+@[\w\.-]+\.\w+$"
    return re.match(regex, email) is not None



#===========================================
# 📁 Configuration Uploads
#===========================================
UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# -------------------------
# 🗄️ Connexion MongoDB
# -------------------------
mongo_client = MongoClient("mongodb://localhost:27017/")
db = mongo_client["chatbot_sante"]
users_collection = db["users"]
conversations_collection = db["conversations"]
consultations_collection = db['consultations']
reminders_collection = db["reminders"]
email_logs_collection = db["email_logs"]
reset_tokens_collection = db["reset_tokens"]
messages_collection = db["messages_privés"]
# ===========================================


# ------------------------------------------------------------
# 🧩 DÉCORATEUR POUR CONTRÔLER LES RÔLES
# ------------------------------------------------------------
def role_required(*roles):
    """Décorateur pour restreindre l’accès à certaines routes selon le rôle."""
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            if 'username' not in session:
                flash("Veuillez vous connecter pour accéder à cette page.")
                return redirect(url_for('login'))
            user_role = session.get('role', 'patiente')
            if user_role not in roles:
                flash("Accès refusé : vous n'avez pas les droits nécessaires.")
                return redirect(url_for('index'))
            return f(*args, **kwargs)
        return wrapper
    return decorator

# -------------------------
# 🔤 Ressources NLTK
# -------------------------
try:
    _ = stopwords.words("french")
    _ = wordnet.synsets("test")
except Exception:
    nltk.download("stopwords")
    nltk.download("wordnet")
    nltk.download("omw-1.4")
    nltk.download("punkt")

lemmatizer = WordNetLemmatizer()
french_stopwords = set(stopwords.words("french"))

# -------------------------
# 🧹 Fonction de nettoyage du texte
# -------------------------
def clean_text(text):
    """Nettoyage basique du texte."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
    text = re.sub(r"[^a-z0-9'\-\s]", " ", text)
    tokens = [t for t in re.split(r"\s+", text) if t and t not in french_stopwords]
    lemma = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(lemma).strip()


# ===========================================
# 📧 TEMPLATE HTML POUR EMAILS
# ===========================================
def email_template(title, body):
    return f"""
    <div style='background:#f7f7f7;padding:20px;'>
        <div style='max-width:600px;margin:auto;background:white;padding:20px;border-radius:10px;border:1px solid #eee;'>
            <h2 style='color:#4CAF50;text-align:center;margin-bottom:20px;'>{title}</h2>
            <div style='font-size:16px;color:#333;line-height:1.7;'>
                {body}
            </div>
            <hr style='margin-top:30px;'>
            <p style='text-align:center;color:#777;font-size:12px;'>
                Plateforme de santé maternelle – © 2025<br>
                Ceci est un message automatique. Merci de ne pas répondre.
            </p>
        </div>
    </div>
    """



# -------------------------
# 📧 Fonction d’envoi d’email
# -------------------------
def send_email(to, subject, content):
    try:
        msg = Message(subject, recipients=[to])
        msg.html = content
        mail.send(msg)
        print(f"[EMAIL OK] → {to}")

        # Log en cas de succès
        log_email(to, subject, content, status="sent")

    except Exception as e:
        print(f"[EMAIL ERROR] {e}")

        # Log en cas d’erreur
        log_email(to, subject, content, status="failed", error_message=str(e))


# ===========================================
# 📊 Fonction d’enregistrement des logs Email
# ===========================================
def log_email(to, subject, content, status="sent", error_message=None):
    email_logs_collection.insert_one({
        "to": to,
        "subject": subject,
        "content": content,
        "status": status,
        "error": error_message,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
# ===========================================

# -------------------------
# 📚 Chargement du dataset
# -------------------------
df = pd.read_csv("training_data.csv")
df.dropna(inplace=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

questions = [clean_text(q) for q in df["question"].tolist()]
reponses = df["reponse"].tolist()

label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(reponses)

# -------------------------
# 🔡 Tokenizer et TF-IDF
# -------------------------
tokenizer_path = "tokenizer.pkl"
if os.path.exists(tokenizer_path):
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
else:
    tokenizer = keras.preprocessing.text.Tokenizer(oov_token="<OOV>")
    tokenizer.fit_on_texts(questions)
    with open(tokenizer_path, "wb") as f:
        pickle.dump(tokenizer, f)

train_sequences = tokenizer.texts_to_sequences(questions)
max_len = max(len(s) for s in train_sequences) if train_sequences else 20
train_sequences = keras.preprocessing.sequence.pad_sequences(train_sequences, maxlen=max_len, padding="post")

tfidf_path = "tfidf.pkl"
if os.path.exists(tfidf_path):
    with open(tfidf_path, "rb") as f:
        tfidf = pickle.load(f)
else:
    tfidf = TfidfVectorizer()
    tfidf.fit(questions)
    with open(tfidf_path, "wb") as f:
        pickle.dump(tfidf, f)

tfidf_matrix = tfidf.transform(questions)

# -------------------------
# 🧠 Construction du modèle
# -------------------------
def Construction_Model(vocab_size, embed_dim=128, input_length=100, n_classes=None):
    inputs = layers.Input(shape=(input_length,))
    x = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, input_length=input_length)(inputs)
    x = layers.LSTM(128)(x)
    outputs = layers.Dense(n_classes, activation='softmax')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

model_path = "chatbot_model.h5"
label_encoder_path = "label_encoder.pkl"

if os.path.exists(model_path) and os.path.exists(label_encoder_path):
    print("Chargement du modèle existant...")
    model = keras.models.load_model(model_path)
    with open(label_encoder_path, "rb") as f:
        label_encoder = pickle.load(f)
else:
    print("🔧 Entraînement du modèle (nouveau)...")
    vocab_size = len(tokenizer.word_index) + 1
    model = Construction_Model(vocab_size=vocab_size, input_length=max_len, n_classes=len(label_encoder.classes_))
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        keras.callbacks.ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss', verbose=1)
    ]
    history = model.fit(
        train_sequences,
        encoded_labels,
        epochs=50,
        batch_size=8,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )
    with open(label_encoder_path, "wb") as f:
        pickle.dump(label_encoder, f)
    # Sauvegarde du tokenizer et du TF-IDF
    with open(tokenizer_path, "wb") as f:
        pickle.dump(tokenizer, f)
    with open(tfidf_path, "wb") as f:
        pickle.dump(tfidf, f)
    # Log d'entraînement
    try:
        pd.DataFrame(history.history).to_csv("training_log.csv", index=False)
    except Exception:
        pass

# -------------------------
# 💬 Génération de réponse
# -------------------------
def semantic_fallback(user_text, threshold=0.75):
    cleaned = clean_text(user_text)
    v = tfidf.transform([cleaned])
    sims = cosine_similarity(v, tfidf_matrix).flatten()
    best_idx = int(np.argmax(sims))
    best_score = float(sims[best_idx])
    if best_score >= threshold:
        return reponses[best_idx], best_score
    return None, best_score


# ===========================================
# 🩺 RÈGLES MÉDICALES — ALERTES AUTOMATIQUES
# ===========================================
pregnancy_alerts = {
    "saignement": "Les saignements doivent être pris au sérieux. Consulte immédiatement un professionnel de santé.",
    "fievre": "Une fièvre élevée peut être dangereuse pendant la grossesse. Contactez votre conseiller.",
    "contraction forte": "Des contractions intenses avant 37 semaines peuvent indiquer un travail prématuré.",
    "douleur abdominale": "Des douleurs abdominales importantes nécessitent un contrôle médical.",
}

weekly_alerts = {
    12: "N'oubliez pas l'échographie du premier trimestre.",
    20: "Pensez à l'échographie morphologique.",
    28: "C'est le moment de vérifier votre taux de fer.",
    36: "Préparez votre plan d’accouchement.",
}
# -------------------------

def generate_response(user_text):
    """
    Génère une réponse intelligente : vérifie d’abord s’il s’agit d’une demande liée aux rappels (vaccins, rdv...),
    sinon utilise le modèle / TF-IDF pour générer la réponse habituelle.
    """
    cleaned = clean_text(user_text)
    username = session.get('username')

    # ---------------------------------------------------
    # 🧭 1️⃣ Détection de requêtes liées aux rappels
    # ---------------------------------------------------
    if any(k in cleaned for k in ["rappel", "vaccin", "rdv", "rendez", "consultation"]):
        if username:
            today = date.today().isoformat()

            # Si la phrase contient "ajoute" ou "crée" → on ajoute un rappel
            if any(k in cleaned for k in ["ajoute", "creer", "crée", "planifie"]):
                # (version simple : on demande à l'utilisateur de préciser ensuite via la page reminders)
                return ("Pour ajouter un rappel, rends-toi sur la page  <b>Mes rappels</b> dans le menu. "
                        "Tu pourras y définir le type, la date et l’heure."), 1.0, "reminder"

            # Sinon, on affiche les rappels à venir
            upcoming = list(reminders_collection.find({
                "username": username,
                "date": {"$gte": today}
            }).sort("date", 1))

            if upcoming:
                responses = [
                    f" {r['type'].capitalize()} prévu le {r['date']} à {r['time']}"
                    for r in upcoming
                ]
                return ("Voici vos rappels à venir :<br>" + "<br>".join(responses)), 1.0, "reminder"
            else:
                return ("Vous n’avez aucun rappel prévu pour le moment."), 1.0, "reminder"


        # ---------------------------------------------------
    # 🩺  ALERTES SELON LES SYMPTÔMES
    # ---------------------------------------------------
    for symptom, alert_msg in pregnancy_alerts.items():
        if symptom in cleaned:
            return (f"<b>Alerte :</b> {alert_msg}", 1.0, "medical_alert")

    # ---------------------------------------------------
    # 📅 ALERTES SELON LA SEMAINE DE GROSSESSE
    # ---------------------------------------------------
    if username:
        user = users_collection.find_one({"username": username})
        profile = user.get("profile", {})

        if profile and profile.get("pregnancy_week"):
            try:
                week = int(profile["pregnancy_week"])
                if week in weekly_alerts:
                    return (weekly_alerts[week], 1.0, "weekly_alert")
            except:
                pass

    # ---------------------------------------------------
    # 🧠 2️⃣ Sinon, comportement normal (ton code d’origine)
    # ---------------------------------------------------
    sem_res, sem_score = semantic_fallback(user_text, threshold=0.66)
    if sem_res:
        return sem_res, sem_score, "tfidf"

    seq = tokenizer.texts_to_sequences([cleaned])
    seq = keras.preprocessing.sequence.pad_sequences(seq, maxlen=max_len, padding="post")
    pred = model.predict(seq)
    pred_label = int(np.argmax(pred, axis=1)[0])
    confidence = float(np.max(pred))

    if confidence < 0.45:
        return "Je ne suis pas sûr de bien comprendre, peux-tu reformuler ?", confidence, "low_conf"
    else:
        return str(label_encoder.inverse_transform([pred_label])[0]), confidence, "model"


# ============================================================
# 🌿 PAGE D’ACCUEIL
# ============================================================
@app.route('/')
def root_redirect():
    return redirect(url_for('home'))

# ============================================================
# 🏠 PAGE D'ACCUEIL
@app.route('/home')
def home():
    if 'username' in session:
        return redirect(url_for('index'))
    return render_template('home.html')


# ============================================================
# 👥 AUTHENTIFICATION UTILISATEUR + STOCKAGE DES CONVERSATIONS
# ============================================================
# ============================================================
# 👥 INSCRIPTION UTILISATEUR
# ============================================================
@app.route('/register', methods=['GET', 'POST'])
def register():
    message = ""
    email = None
    default_role = "patiente"
    current_role = session.get('role')

    if request.method == 'POST':
        nom = request.form.get('nom')
        prenom = request.form.get('prenom')
        username = request.form.get('username')
        password = request.form.get('password')
        email = request.form.get('email')


        # Rôle selon admin ou non
        role_from_form = request.form.get('role') if current_role == 'admin' else None
        role = role_from_form if role_from_form else default_role

        # Vérification username
        if users_collection.find_one({'username': username}):
            message = "Ce nom d'utilisateur existe déjà."
        else:
            hashed_password = generate_password_hash(password)

            # Photo
            uploaded_photo = request.files.get('photo')
            photo_filename = "image.png"
            if uploaded_photo:
                photo_filename = username + "_" + secure_filename(uploaded_photo.filename)
                uploaded_photo.save("static/uploads/" + photo_filename)

            # 🎯 NOUVEAU : semaine de grossesse
            pregnancy_week = request.form.get("pregnancy_week")

            expected_due = ""
            if pregnancy_week and pregnancy_week.isdigit():
                remaining = 40 - int(pregnancy_week)
                expected_due = (date.today() + timedelta(weeks=remaining)).strftime("%Y-%m-%d")

            profile = {
                "dob": request.form.get("dob"),
                "pregnancy_week": pregnancy_week,
                "expected_due": expected_due,
                "num_children": request.form.get("num_children"),
                "allergies": request.form.get("allergies"),
                "chronic_conditions": request.form.get("chronic_conditions"),
                "current_medications": request.form.get("current_medications"),
                "blood_type": request.form.get("blood_type"),
                "emergency_contact": {
                    "name": request.form.get("emergency_name"),
                    "phone": request.form.get("emergency_phone")
                }
            }

            users_collection.insert_one({
                "nom": nom,
                "prenom": prenom,
                "username": username,
                "password": hashed_password,
                "role": role,
                "email": email,
                "photo": photo_filename,
                "profile": profile
            })

            # Email de confirmation
            try:
                subject = "Bienvenue sur votre espace santé maternelle"
                content = f"""
                <h2>Bienvenue {prenom} 👋</h2>
                <p>Votre compte a été créé avec succès.</p>
                <p><strong>Nom d’utilisateur :</strong> {username}</p>
                <p>Vous pouvez maintenant vous connecter à la plateforme.</p>
                <br>
                <p style='color:#777'>Merci de faire confiance à notre plateforme.</p>
                """

                send_email(email, subject, content)
            except:
                pass


            return redirect(url_for("login"))

    roles_options = ['patiente', 'conseiller', 'admin'] if current_role == 'admin' else None
    return render_template("register.html", message=message, roles_options=roles_options)


# ============================================================
# 👥 CONNEXION UTILISATEUR
# ============================================================
@app.route('/login', methods=['GET', 'POST'])
def login():
    message = ""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        user = users_collection.find_one({'username': username})
        if user and check_password_hash(user['password'], password):
            session['username'] = username
            session['role'] = user.get('role', 'patiente')  # rôle stocké en session

            # Redirection selon le rôle :
            role = session['role']
            if role == 'admin':
                return redirect(url_for('admin'))
            elif role == 'patiente':
                return redirect(url_for('index'))
            elif role == 'conseiller':
                return redirect(url_for("conseiller_dashboard"))
            else:
                return redirect(url_for('home'))
        else:
            message = "Nom d'utilisateur ou mot de passe incorrect."
    return render_template('login.html', message=message)


# ============================================================
# 👥 RÉINITIALISATION DU MOT DE PASSE
# ============================================================
@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    message = ""

    if request.method == 'POST':
        email = request.form.get('email')

        user = users_collection.find_one({"email": email})
        if not user:
            message = "❌ Aucun compte trouvé avec cet email."
        else:
            # Création d'un token sécurisé unique
            token = os.urandom(24).hex()
            expiration = datetime.now() + timedelta(minutes=15)

            reset_tokens_collection.insert_one({
                "email": email,
                "token": token,
                "expires_at": expiration
            })

            reset_link = url_for('reset_password', token=token, _external=True)

            # 📧 Email HTML
            subject = "🔐 Réinitialisation de votre mot de passe"
            content = email_template(
                "Réinitialisation de votre mot de passe",
                f"""
                <p>Bonjour,</p>
                <p>Vous avez demandé à réinitialiser votre mot de passe.</p>
                <p>Cliquez sur le lien ci-dessous :</p>
                <p><a href='{reset_link}' style='color:#4CAF50;font-weight:bold;'>Réinitialiser mon mot de passe</a></p>
                <p>Ce lien expire dans 15 minutes.</p>
                """
            )

            send_email(email, subject, content)
            message = "📩 Un lien de réinitialisation a été envoyé à votre email."

    return render_template("forgot_password.html", message=message)

#===========================================================
# 👥 RÉINITIALISATION DU MOT DE PASSE VIA LE LIEN EMAIL
#===========================================================
@app.route('/reset_password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    message = ""

    token_data = reset_tokens_collection.find_one({"token": token})

    if not token_data:
        return render_template("reset_failure.html", email=None)

    # Vérifier expiration
    if datetime.now() > token_data["expires_at"]:
        reset_tokens_collection.delete_one({"token": token})
        return render_template("reset_failure.html", email=token_data["email"])

    if request.method == 'POST':
        new_pass = request.form.get('password')
        confirm_pass = request.form.get('confirm_password')

        if new_pass != confirm_pass:
            message = "❌ Les mots de passe ne correspondent pas."
        else:
            hashed = generate_password_hash(new_pass)

            # Mise à jour du mot de passe
            users_collection.update_one(
                {"email": token_data["email"]},
                {"$set": {"password": hashed}}
            )

        # Suppression du token pour sécurité
            reset_tokens_collection.delete_one({"token": token})

        # ============================================
        # 📧 EMAIL DE CONFIRMATION APRÈS MODIFICATION
        # ============================================
            subject = "🔐 Votre mot de passe a été modifié"
            content = email_template(
                "Mot de passe modifié",
                 f"""
                <p>Bonjour,</p>
                <p>Votre mot de passe a été modifié avec succès.</p>
                <p>Si vous n'êtes pas à l'origine de cette action, veuillez contacter le support immédiatement.</p>
                <br>
                <p>Merci d'utiliser notre plateforme de santé.</p>
                """
            )

            send_email(token_data["email"], subject, content)

            return render_template("reset_success.html", email=token_data["email"])


    return render_template("reset_password.html", message=message)


# ============================================================
# 👥 DÉCONNEXION UTILISATEUR
# ============================================================
@app.route('/logout')
def logout():
    session.pop('username', None)
    session.pop('role', None)
    return redirect(url_for('home'))

# ============================================================
# 👨‍💼 ROUTES ADMIN — GESTION DES UTILISATEURS
# ============================================================

def admin_required(f):
    """Décorateur pour bloquer l’accès aux non-admins."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session or session.get('role') != 'admin':
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function


@app.route('/admin')
@admin_required
def admin():
    total_patients = users_collection.count_documents({"role": "patiente"})
    total_conseillers = users_collection.count_documents({"role": "conseiller"})
    total_consultations = consultations_collection.count_documents({})
    
    # Rappels à partir d'aujourd'hui
    from datetime import date
    today = date.today().isoformat()
    total_reminders = reminders_collection.count_documents({"date": {"$gte": today}})

    return render_template(
        "admin.html",
        total_patients=total_patients,
        total_conseillers=total_conseillers,
        total_consultations=total_consultations,
        total_reminders=total_reminders
    )

# ============================================================
# 👨‍💼 GESTION DES UTILISATEURS PAR L'ADMIN
# ============================================================
@app.route('/admin/users')
def manage_users():
    if 'username' not in session or session['role'] != 'admin':
        return redirect(url_for('login'))

    users = list(users_collection.find())
    return render_template('admin_users.html', users=users)

# ============================================================
# 👨‍💼 GESTION DES CONSULTATIONS PAR LE CONSEILLER
# ============================================================
@app.route("/conseiller/consultations", methods=["GET", "POST"])
def conseiller_consultations():
    if session.get("role") != "conseiller":
        return redirect(url_for("login"))

    username = session.get("username")

    # 🔥 Récupération du conseiller connecté
    user = users_collection.find_one({"username": username})

    # =====================================
    # 🔥 TRAITEMENT DU POST (APPROUVER / REJETER)
    # =====================================
    if request.method == "POST":
        consult_id = request.form.get("consult_id")
        action = request.form.get("action")

        if consult_id and action:

            # ✔ APPROUVER
            if action == "approve":
                consultations_collection.update_one(
                    {"_id": ObjectId(consult_id)},
                    {"$set": {"status": "approuvee"}}
                )

            # ❌ REJETER
            elif action == "reject":
                consultations_collection.update_one(
                    {"_id": ObjectId(consult_id)},
                    {"$set": {"status": "rejettee"}}
                )

        # 🔁 Recharge la page après action
        return redirect(url_for("conseiller_consultations"))

    # =====================================
    # 🔥 LECTURE DES DONNÉES (GET)
    # =====================================
    consultations_attente = list(consultations_collection.find({
        "conseiller": username,
        "status": "en_attente"
    }))

    consultations_ok = list(consultations_collection.find({
        "conseiller": username,
        "status": "approuvee"
    }))

    consultations_rejet = list(consultations_collection.find({
        "conseiller": username,
        "status": "rejettee"
    }))

    return render_template(
        "conseiller_consultations.html",
        user=user,
        consultations_attente=consultations_attente,
        consultations_ok=consultations_ok,
        consultations_rejet=consultations_rejet
    )



# ============================================================
# 👨‍💼 GESTION DES CONSULTATIONS PAR L'ADMIN
# ============================================================
@app.route('/admin/consultations')
def admin_consultations():
    if 'username' not in session or session['role'] != 'admin':
        return redirect(url_for('login'))

    consultations = list(consultations_collection.find().sort("date", -1))
    return render_template('admin_consultations.html', consultations=consultations)

# ============================================================
# 👨‍💼 GESTION DES RAPPELS PAR L'ADMIN
# ============================================================
@app.route('/admin/reminders')
def admin_reminders():
    if 'username' not in session or session['role'] != 'admin':
        return redirect(url_for('login'))

    reminders = list(reminders_collection.find().sort("date", 1))
    return render_template('admin_reminders.html', reminders=reminders)

# ============================================================
# 👨‍💼 SUPPRESSION D'UN RAPPEL PAR L'ADMIN
# ============================================================
@app.route('/admin/delete_reminder/<reminder_id>')
def delete_reminder(reminder_id):
    if 'username' not in session or session['role'] != 'admin':
        return redirect(url_for('login'))

    reminders_collection.delete_one({"_id": ObjectId(reminder_id)})
    return redirect(url_for('admin_reminders'))


# ============================================================
# 👨‍💼 AJOUT D'UN CONSEILLER PAR L'ADMIN
# ============================================================
@app.route('/admin/add_conseiller', methods=['GET', 'POST'])
@admin_required
def add_conseiller():
    """Formulaire d’ajout de conseiller"""
    message = ""
    if request.method == 'POST':
        nom = request.form.get('nom')
        prenom = request.form.get('prenom')
        username = request.form.get('username')
        password = request.form.get('password')
        email = request.form.get('email')   # 👈 NOUVEAU

        # Vérification username
        if users_collection.find_one({'username': username}):
            message = "⚠️ Ce nom d'utilisateur existe déjà."
        else:
            hashed_password = generate_password_hash(password)

            users_collection.insert_one({
                'nom': nom,
                'prenom': prenom,
                'username': username,
                'password': hashed_password,
                'role': 'conseiller',
                'email': email,            # 👈 AJOUT OBLIGATOIRE
                'verified': True
            })

            # 📧 ENVOI AUTOMATIQUE EMAIL DE BIENVENUE AU CONSEILLER
            try:
                subject = "Bienvenue dans la plateforme en tant que Conseiller"
                content = email_template(
                    "Bienvenue Conseiller",
                    f"""
                    <p>Bonjour <strong>{prenom}</strong>,</p>
                    <p>Votre compte Conseiller a été créé avec succès.</p>
                    <p>Nom d’utilisateur : <strong>{username}</strong></p>
                    <p>Vous pouvez vous connecter pour gérer les consultations des patientes.</p>
                    """
                )
                send_email(email, subject, content)
            except:
                pass

            message = "✅ Conseiller ajouté et email envoyé."

    return render_template('add_conseiller.html', message=message)

# ============================================================
# 👨‍💼 DISCUSSIONS PRIVÉES PAR LE CONSEILLER
# ============================================================
@app.route("/conseiller/discussions")
def conseiller_discussions():
    if session.get("role") != "conseiller":
        return redirect(url_for("login"))

    username = session.get("username")

    # 🔥 Récupération du conseiller connecté
    user = users_collection.find_one({"username": username})

    # 🔥 Récupérer les consultations approuvées
    consultations_ok = list(consultations_collection.find({
        "conseiller": username,
        "status": "approuvee"
    }))

    return render_template(
        "conseiller_discussions.html",
        user=user,                    # 🔥 obligatoire !
        consultations_ok=consultations_ok
    )



# ============================================================
# 👨‍💼 DASHBOARD CONSEILLER
# ============================================================
@app.route('/conseiller/dashboard')
def conseiller_dashboard():
    if 'username' not in session or session.get('role') != 'conseiller':
        return redirect(url_for('login'))

    username = session['username']

    total_patients = users_collection.count_documents({"role": "patiente"})
    total_consultations = consultations_collection.count_documents({"conseiller": username})

    return render_template(
        "conseiller_dashboard.html",
        username=username,
        total_patients=total_patients,
        total_consultations=total_consultations
    )

# ============================================================
# 👨‍💼 LISTE DES PATIENTES POUR LE CONSEILLER
# ============================================================
@app.route('/conseiller/patients')
def conseiller_patients():
    if 'username' not in session or session.get('role') != 'conseiller':
        return redirect(url_for('login'))

    patients = list(users_collection.find({"role": "patiente"}))
    return render_template('conseiller_patients.html', patients=patients)

# ============================================================
# 👨‍💼 PROFIL DU CONSEILLER
# ============================================================
@app.route('/conseiller/profile')
def conseiller_profile():
    if 'username' not in session or session.get('role') != 'conseiller':
        return redirect(url_for('login'))

    user = users_collection.find_one({"username": session["username"]})
    return render_template("conseiller_profile.html", user=user)

# ============================================================
# 👨‍💼 MODIFICATION DU PROFIL DU CONSEILLER
# ============================================================
@app.route('/conseiller/profile/edit', methods=['GET', 'POST'])
def conseiller_profile_edit():
    if 'username' not in session or session.get('role') != 'conseiller':
        return redirect(url_for('login'))

    username = session["username"]
    user = users_collection.find_one({"username": username})
    message = ""

    if request.method == 'POST':

        # photo update
        if 'photo' in request.files:
            photo = request.files.get("photo")
            if photo:
                filename = username + "_" + secure_filename(photo.filename)
                photo.save("static/uploads/" + filename)
                users_collection.update_one({"username": username}, {"$set": {"photo": filename}})
                message = "Photo mise à jour avec succès."

        # data update
        nom = request.form.get("nom")
        prenom = request.form.get("prenom")
        email = request.form.get("email")

        users_collection.update_one(
            {"username": username},
            {"$set": {"nom": nom, "prenom": prenom, "email": email}}
        )

        message = "Informations mises à jour avec succès."

        # recharger
        user = users_collection.find_one({"username": username})

    photo_url = url_for('static', filename='uploads/' + user.get("photo", "image.png"))
    return render_template("conseiller_profile_edit.html", user=user, message=message, photo_url=photo_url)


# ============================================================
# 👨‍💼 LISTE DES PATIENTES POUR L'ADMIN
# ============================================================
@app.route('/admin/list_patients')
@admin_required
def list_patients():
    """Liste tous les utilisateurs avec rôle = patiente"""
    patients = list(users_collection.find({'role': 'patiente'}))
    return render_template('list_patients.html', patients=patients)

# ============================================================
# 👨‍💼 LISTE DES CONSEILLERS POUR L'ADMIN
# ============================================================
@app.route('/admin/list_conseillers')
@admin_required
def list_conseillers():
    """Liste tous les utilisateurs avec rôle = conseiller"""
    conseillers = list(users_collection.find({'role': 'conseiller'}))
    return render_template('list_conseillers.html', conseillers=conseillers)

# ============================================================
# 👨‍💼 SUPPRESSION D'UN UTILISATEUR PAR L'ADMIN
# ============================================================
@app.route('/admin/delete_user/<user_id>')
@admin_required
def delete_user(user_id):
    """Supprimer un utilisateur"""
    users_collection.delete_one({'_id': ObjectId(user_id)})
    return redirect(url_for('admin'))



# ============================================================
# ✏️ MODIFICATION DES INFORMATIONS UTILISATEUR
# ============================================================
@app.route('/admin/edit_user/<user_id>', methods=['GET', 'POST'])
@admin_required
def edit_user(user_id):
    """Modifier les informations d’un utilisateur"""
    user = users_collection.find_one({'_id': ObjectId(user_id)})
    if not user:
        return redirect(url_for('admin'))

    message = ""

    if request.method == 'POST':
        nom = request.form.get('nom')
        prenom = request.form.get('prenom')
        username = request.form.get('username')
        role = request.form.get('role')

        users_collection.update_one(
            {'_id': ObjectId(user_id)},
            {'$set': {
                'nom': nom,
                'prenom': prenom,
                'username': username,
                'role': role
            }}
        )
        message = "✅ Informations mises à jour avec succès !"
        return redirect(url_for('admin'))

    return render_template('edit_user.html', user=user, message=message)


@app.route('/admin/email_logs')
def email_logs():
    if 'username' not in session or session.get('role') != 'admin':
        return redirect(url_for('login'))

    page = int(request.args.get("page", 1))
    per_page = 20

    search = request.args.get("search", "").strip()
    filter_status = request.args.get("status", "").strip()

    query = {}
    if search:
        query["$or"] = [
            {"to": {"$regex": search, "$options": "i"}},
            {"subject": {"$regex": search, "$options": "i"}},
        ]

    if filter_status in ["sent", "failed"]:
        query["status"] = filter_status

    # Pagination
    total = email_logs_collection.count_documents(query)
    logs = (email_logs_collection
            .find(query)
            .sort("timestamp", -1)
            .skip((page - 1) * per_page)
            .limit(per_page))

    total_pages = (total + per_page - 1) // per_page

    return render_template(
        "email_logs.html",
        logs=logs,
        page=page,
        total_pages=total_pages,
        search=search,
        filter_status=filter_status
    )



# ============================================================
# ⏰ GESTION DES RAPPELS MÉDICAUX
# ============================================================
@app.route('/reminders', methods=['GET', 'POST'])
def reminders():
    """Page pour gérer, ajouter, modifier et supprimer les rappels (vaccins, rendez-vous, etc.)."""
    if 'username' not in session or session.get('role') != 'patiente':
        return redirect(url_for('login'))

    username = session['username']
    message = ""

    # ----- AJOUT D'UN RAPPEL -----
    if request.method == 'POST' and 'add' in request.form:
        reminder_type = request.form.get('type')
        reminder_message = request.form.get('message')
        reminder_date = request.form.get('date')
        reminder_time = request.form.get('time')

        if reminder_date and reminder_time and reminder_message:
            reminders_collection.insert_one({
                "username": username,
                "type": reminder_type,
                "message": reminder_message,
                "date": reminder_date,
                "time": reminder_time,
                "notified": False
            })
            message = "✅ Rappel ajouté avec succès !"
        else:
            message = "❌ Veuillez remplir tous les champs."

    # ----- SUPPRESSION D'UN RAPPEL -----
    if request.method == 'POST' and 'delete_id' in request.form:
        delete_id = request.form.get('delete_id')
        if delete_id:
            reminders_collection.delete_one({
                "_id": ObjectId(delete_id),
                "username": username
            })
            message = "🗑️ Rappel supprimé avec succès."

    # ----- MODIFICATION D'UN RAPPEL -----
    if request.method == 'POST' and 'edit_id' in request.form:
        edit_id = request.form.get('edit_id')
        new_type = request.form.get('type')
        new_message = request.form.get('message')
        new_date = request.form.get('date')
        new_time = request.form.get('time')

        if edit_id and new_message and new_date and new_time:
            reminders_collection.update_one(
                {"_id": ObjectId(edit_id), "username": username},
                {"$set": {
                    "type": new_type,
                    "message": new_message,
                    "date": new_date,
                    "time": new_time
                }}
            )
            message = "✏️ Rappel modifié avec succès."

    # ----- LISTER LES RAPPELS -----
    user_reminders = list(reminders_collection.find({"username": username}).sort("date", 1))
    return render_template('reminders.html', username=username, reminders=user_reminders, message=message)


# ============================
# 👩‍⚕️ PROFIL MÉDICAL PATIENT
# ============================
from datetime import datetime, timedelta
@app.route("/profile")
def profile():
    if "username" not in session:
        return redirect("/login")

    username = session["username"]
    user = users_collection.find_one({"username": username})
    profile = user.get("profile", {})

    # Contact d'urgence
    emergency = profile.get("emergency_contact", {"name": "", "phone": ""})

    # 🎯 Charger la semaine de grossesse
    pregnancy_week = None
    if profile.get("pregnancy_week"):
        try:
            pregnancy_week = int(profile["pregnancy_week"])
        except:
            pregnancy_week = None

    return render_template(
        "profile.html",
        user=user,
        profile=profile,
        emergency=emergency,
        pregnancy_week=pregnancy_week
    )


# ============================================================
# ✏️ MODIFICATION DU PROFIL MÉDICAL PATIENT
# ============================================================
# ============================================================
# ✏️ MODIFICATION DU PROFIL MÉDICAL PATIENT
# ============================================================
@app.route('/profile/edit', methods=["GET", "POST"])
def profile_edit():
    if "username" not in session:
        return redirect(url_for("login"))

    username = session["username"]
    user = users_collection.find_one({"username": username})
    profile = user.get("profile", {})
    message = ""

    # --- Mise à jour photo ---
    if "update_photo" in request.form:
        photo = request.files.get("photo")
        if photo:
            filename = username + "_" + photo.filename
            photo.save("static/uploads/" + filename)
            users_collection.update_one(
                {"username": username},
                {"$set": {"photo": filename}}
            )
            message = "Photo mise à jour."

    # --- Mise à jour infos ---
    if "update_profile" in request.form:

        pregnancy_week = request.form.get("pregnancy_week")

        expected_due = ""
        if pregnancy_week and pregnancy_week.isdigit():
            remaining = 40 - int(pregnancy_week)
            expected_due = (date.today() + timedelta(weeks=remaining)).strftime("%Y-%m-%d")

        new_profile = {
            "dob": request.form.get("dob"),
            "pregnancy_week": pregnancy_week,
            "expected_due": expected_due,
            "num_children": request.form.get("num_children"),
            "allergies": request.form.get("allergies"),
            "current_medications": request.form.get("current_medications"),
            "blood_type": request.form.get("blood_type"),
            "emergency_contact": {
                "name": request.form.get("emergency_name"),
                "phone": request.form.get("emergency_phone")
            }
        }

        users_collection.update_one(
            {"username": username},
            {"$set": {"profile": new_profile}}
        )
        message = "Profil mis à jour."

    # Rechargement
    user = users_collection.find_one({"username": username})
    profile = user.get("profile", {})
    photo_url = url_for("static", filename="uploads/" + user.get("photo"))

    return render_template(
        "profile_edit.html",
        user=user,
        profile=profile,
        photo_url=photo_url,
        message=message
    )


# ============================================================



# Admin : voir le profil d'un utilisateur
@app.route('/admin/view_profile/<user_id>')
@admin_required
def admin_view_profile(user_id):
    user = users_collection.find_one({'_id': ObjectId(user_id)})
    if not user:
        return redirect(url_for('admin'))
    return render_template('admin_view_profile.html', user=user)




# ============================================================
# 💬 ROUTE PRINCIPALE DU CHATBOT
# ============================================================
def fetch_last_messages(username, limit=100):
    docs = list(conversations_collection.find({'username': username}, {'_id': 0}).sort([('_id', -1)]).limit(limit))
    docs.reverse()
    return docs
@app.route('/index', methods=['GET', 'POST'])
def index():
    if 'username' not in session:
        return redirect(url_for('login'))

    if session.get('role') != 'patiente':
        return redirect(url_for('home'))

    username = session['username']
    user = users_collection.find_one({"username": username})

    # --- Récupérer toutes les conversations du patient ---
    all_conversations = list(conversations_collection.find({"username": username}))

    # --- Si aucune conversation n'existe, on en crée une nouvelle ---
    if not all_conversations:
        first_conv_id = conversations_collection.insert_one({
            "username": username,
            "name": "Conversation 1",
            "messages": [],
            "created_at": datetime.now()
        }).inserted_id

        session["current_conversation"] = str(first_conv_id)

    # --- Sinon si aucune conversation active n'est sélectionnée ---
    if "current_conversation" not in session:
        last_conv = all_conversations[-1]
        session["current_conversation"] = str(last_conv["_id"])

    # --- Charger conversation active ---
    conv_id = session["current_conversation"]
    active_conv = conversations_collection.find_one({"_id": ObjectId(conv_id)})

    #
    search = request.args.get("q", "")

    if search:
      conversations = list(conversations_collection.find({
        "username": username,
        "name": {"$regex": search, "$options": "i"}
    }))
    else:
       conversations = list(conversations_collection.find({"username": username}))


    # --- GESTION DU MESSAGE ENVOYÉ ---
    if request.method == "POST":
        user_msg = request.form.get("input", "").strip()

        if user_msg:
            # ⬅ Enregistrer le message utilisateur
            conversations_collection.update_one(
                {"_id": ObjectId(conv_id)},
                {"$push": {"messages": {"sender": "user", "text": user_msg}}}
            )

            # ⬅ Nouvelle réponse du bot
            bot_reply, score, source = generate_response(user_msg)

            conversations_collection.update_one(
                {"_id": ObjectId(conv_id)},
                {"$push": {"messages": {
                    "sender": "bot",
                    "text": bot_reply,
                    "score": score,
                    "source": source
                }}}
            )

        # recharger après insertion
        active_conv = conversations_collection.find_one({"_id": ObjectId(conv_id)})

    return render_template(
        "index.html",
        username=username,
        user=user,
        conversations=all_conversations,
        active_conv=active_conv,
        messages=active_conv["messages"]
    )


# ============================================================
# ➕ CRÉATION D'UNE NOUVELLE CONVERSATION
# ============================================================
@app.route('/new_conversation')
def new_conversation():
    if "username" not in session:
        return redirect(url_for('login'))

    username = session["username"]

    count = conversations_collection.count_documents({"username": username})
    new_name = f"Conversation {count + 1}"

    conv_id = conversations_collection.insert_one({
        "username": username,
        "name": new_name,
        "messages": [],
        "created_at": datetime.now()
    }).inserted_id

    session["current_conversation"] = str(conv_id)

    return redirect(url_for('index'))

# ============================================================
# 🔄 OUVRIR UNE CONVERSATION EXISTANTE
# ============================================================
@app.route('/open_conversation/<cid>')
def open_conversation(cid):
    session["current_conversation"] = cid
    return redirect(url_for("index"))

# ============================================================
# 🗑️ SUPPRESSION D'UNE CONVERSATION
# ============================================================
@app.route("/delete_conversation")
def delete_conversation():
    cid = request.args.get("cid")

    if not cid:
        return redirect(url_for("index"))

    conversations_collection.delete_one({"_id": ObjectId(cid)})

    # Reprendre la dernière conversation restante
    username = session["username"]
    remaining = list(conversations_collection.find({"username": username}))

    if remaining:
        session["current_conversation"] = str(remaining[-1]["_id"])
    else:
        # Créer une nouvelle conversation vide
        new_id = conversations_collection.insert_one({
            "username": username,
            "title": "Conversation 1",
            "messages": [],
            "created_at": datetime.now()
        }).inserted_id
        session["current_conversation"] = str(new_id)

    return redirect(url_for("index"))


# ============================================================
# ✏️ RENOMMER UNE CONVERSATION
# ============================================================
@app.route("/rename_conversation", methods=["POST"])
def rename_conversation():
    if "username" not in session:
        return redirect(url_for("login"))

    cid = request.form.get("cid")
    new_name = request.form.get("new_name")

    conversations_collection.update_one(
        {"_id": ObjectId(cid), "username": session["username"]},
        {"$set": {"name": new_name}}
    )

    return redirect(url_for("index"))


# ============================================================
# 📊 METRICS & PERFORMANCE
# ============================================================
@app.route('/metrics')
@role_required('admin')
def metrics():
    if os.path.exists("training_log.csv"):
        df = pd.read_csv("training_log.csv")
        html = df.to_html(classes="table table-striped", index=False)
        return render_template('metrics.html', table=html)
    return "Aucun log d'entraînement trouvé."


@app.route('/performance')
@role_required('admin', 'conseiller')
def performance():
    try:
        y_true = encoded_labels
        y_pred_probs = model.predict(train_sequences)
        y_pred = np.argmax(y_pred_probs, axis=1)

        report = classification_report(y_true, y_pred, output_dict=True)
        accuracy = accuracy_score(y_true, y_pred) * 100
        recall = report["weighted avg"]["recall"] * 100
        f1 = report["weighted avg"]["f1-score"] * 100

        return render_template(
            'performance.html',
            accuracy=round(accuracy, 2),
            recall=round(recall, 2),
            f1=round(f1, 2)
        )

    except Exception as e:
        return f"Erreur lors du calcul des performances : {e}"


@app.route('/reset')
def reset():
    username = session.get('username')
    if username:
        conversations_collection.delete_many({'username': username})
    return redirect(url_for('index'))

# ============================================================
# 🩺 DEMANDE DE CONSULTATION PAR UNE PATIENTE
# ============================================================
@app.route('/consultation/request', methods=['GET', 'POST'])
def request_consultation():
    if 'username' not in session or session.get('role') != 'patiente':
        return redirect(url_for('login'))

    username = session['username']
    message = ""

    conseillers = list(users_collection.find({"role": "conseiller"}))

    if request.method == 'POST':
        conseiller = request.form.get('conseiller')
        motif = request.form.get('motif')
        date_consult = request.form.get('date')
        message_patient = request.form.get('message')

        if conseiller and motif and date_consult:
            # Enregistrement en base
            consultations_collection.insert_one({
                "patient": username,
                "conseiller": conseiller,
                "date": date_consult,
                "motif": motif,
                "message": message_patient,
                "status": "en_attente"
            })

            message = "✅ Votre demande de consultation a été envoyée avec succès."

            # -----------------------------------------
            # 📧 Envoi EMAIL AUTOMATIQUE AU CONSEILLER
            # -----------------------------------------
            conseiller_user = users_collection.find_one({"username": conseiller})

            if conseiller_user and conseiller_user.get("email"):
                subject = "📩 Nouvelle demande de consultation"
                content = f"""
                <h2>Nouvelle demande de consultation</h2>
                <p><strong>Patiente :</strong> {username}</p>
                <p><strong>Motif :</strong> {motif}</p>
                <p><strong>Date souhaitée :</strong> {date_consult}</p>
                <p><strong>Message :</strong> {message_patient}</p>
                <br>
                <p>Connectez-vous sur la plateforme pour approuver ou rejeter la demande.</p>
                """

                send_email(conseiller_user["email"], subject, content)

        else:
            message = "❌ Veuillez remplir tous les champs."

    return render_template('request_consultation.html', conseillers=conseillers, message=message)

# ============================================================
# 🩺 LISTE DES DEMANDES DE CONSULTATION D'UNE PATIENTE
# ============================================================
@app.route('/consultation/my_requests')
def my_consultations():
    if 'username' not in session or session.get('role') != 'patiente':
        return redirect(url_for('login'))

    username = session['username']
    demandes = list(consultations_collection.find({"patient": username}).sort("date", -1))
    return render_template('my_consultations.html', demandes=demandes)

# ============================================================
# 🩺 GESTION DES DEMANDES DE CONSULTATION PAR LES CONSEILLERS
# ============================================================
@app.route('/consultation/manage', methods=['GET', 'POST'])
def manage_consultations():
    if 'username' not in session or session.get('role') != 'conseiller':
        return redirect(url_for('login'))

    username = session['username']  # conseiller

    if request.method == 'POST':
        consult_id = request.form.get('consult_id')
        action = request.form.get('action')

        if consult_id and action:

            # Récupération de la consultation
            consult = consultations_collection.find_one({"_id": ObjectId(consult_id)})

            # 🔹 APPROBATION
            if action == "approve":
                heure = request.form.get("heure")  # récupère l'heure envoyée par la popup

                consultations_collection.update_one(
                    {"_id": ObjectId(consult_id)},
                    {"$set": {"status": "approuvee", "heure": heure}}
                )

                # ➕ Ajout automatique au rappel du patient
                reminders_collection.insert_one({
                    "username": consult["patient"],
                    "type": "consultation",
                    "message": f"Consultation approuvée avec {consult['conseiller']} - Motif : {consult['motif']}",
                    "date": consult["date"],
                    "time": heure,
                    "notified": False
                })

                # -----------------------------------------
                # 📧 EMAIL AU PATIENT : CONSULTATION APPROUVÉE
                # -----------------------------------------
                patient_user = users_collection.find_one({"username": consult["patient"]})
                if patient_user and patient_user.get("email"):
                    subject = "✅ Votre consultation a été approuvée"
                    content = f"""
                    <h2>Votre consultation est approuvée !</h2>
                    <p><strong>Conseiller :</strong> {consult['conseiller']}</p>
                    <p><strong>Motif :</strong> {consult['motif']}</p>
                    <p><strong>Date :</strong> {consult['date']} à {heure}</p>
                    <br>
                    <p>Un rappel vous sera envoyé automatiquement.</p>
                    """
                    send_email(patient_user["email"], subject, content)

            # 🔹 REJET
            elif action == "reject":
                consultations_collection.update_one(
                    {"_id": ObjectId(consult_id)},
                    {"$set": {"status": "rejettee"}}
                )

                # -----------------------------------------
                # 📧 EMAIL AU PATIENT : CONSULTATION REJETÉE
                # -----------------------------------------
                patient_user = users_collection.find_one({"username": consult["patient"]})
                if patient_user and patient_user.get("email"):
                    subject = "❌ Votre consultation a été rejetée"
                    content = f"""
                    <h2>Votre demande de consultation a été rejetée</h2>
                    <p><strong>Conseiller :</strong> {consult['conseiller']}</p>
                    <p><strong>Motif :</strong> {consult['motif']}</p>
                    <br>
                    <p>Vous pouvez soumettre une nouvelle demande à tout moment.</p>
                    """
                    send_email(patient_user["email"], subject, content)

    # Liste des demandes à afficher
    demandes_attente = list(consultations_collection.find({
        "conseiller": username,
        "status": "en_attente"
    }).sort("date", -1))

    consultations_ok = list(consultations_collection.find({
        "conseiller": username,
        "status": "approuvee"
    }).sort("date", -1))


    return render_template(
    'manage_consultations.html',
    demandes_attente=demandes_attente,
    consultations_ok=consultations_ok
)


# ============================================================
# 💬 CHAT ENTRE PATIENTE ET CONSEILLER
# ============================================================
@app.route('/chat/<conseiller>', methods=['GET', 'POST'])
def chat_with_conseiller(conseiller):
    if 'username' not in session or session.get('role') != 'patiente':
        return redirect(url_for('login'))

    username = session['username']

    if request.method == 'POST':
        msg = request.form.get('message')
        if msg:
            messages_collection.insert_one({
                "sender": username,
                "receiver": conseiller,
                "message": msg,
                "timestamp": datetime.now(),
                "seen": False
            })

    messages = list(messages_collection.find({
        "$or": [
            {"sender": username, "receiver": conseiller},
            {"sender": conseiller, "receiver": username}
        ]
    }).sort("timestamp", 1))
    # Marquer les messages du conseiller comme "vus"
    messages_collection.update_many(
        {"receiver": session['username'], "sender": conseiller, "seen": False},
        {"$set": {"seen": True}}
    )


    return render_template("chat.html", messages=messages, conseiller=conseiller)

# ============================================================
# 💬 CHAT ENTRE CONSEILLER ET PATIENTE
# ============================================================
@app.route('/chat_patient/<patient>', methods=['GET', 'POST'])
def chat_patient(patient):
    if 'username' not in session or session.get('role') != 'conseiller':
        return redirect(url_for('login'))

    conseiller = session['username']

    if request.method == 'POST':
        msg = request.form.get('message')
        if msg:
            messages_collection.insert_one({
                "sender": conseiller,
                "receiver": patient,
                "message": msg,
                "timestamp": datetime.now(),
                "seen": False
            })

    messages = list(messages_collection.find({
        "$or": [
            {"sender": patient, "receiver": conseiller},
            {"sender": conseiller, "receiver": patient}
        ]
    }).sort("timestamp", 1))
    # Marquer les messages du patient comme "vus"
    messages_collection.update_many(
        {"receiver": session['username'], "sender": patient, "seen": False},
        {"$set": {"seen": True}}
    )


    return render_template("chat_conseiller.html", messages=messages, patient=patient)




# ============================================================
# 🔄 MISE À JOUR AUTOMATIQUE DES STATUTS DE CONSULTATION
# ============================================================
@app.route('/consultation/status')
def get_consultation_status():
    """Retourne les demandes de consultation d’une patiente au format JSON."""
    if 'username' not in session or session.get('role') != 'patiente':
        return {"error": "unauthorized"}, 403
    
    username = session['username']
    demandes = list(consultations_collection.find(
        {"patient": username},
        {"_id": 0}
    ).sort("date", -1))
    
    return {"demandes": demandes}

# ============================================================
# ⏰ VÉRIFICATION DES RAPPELS ET ENVOI D'EMAILS AUTOMATIQUES
# ============================================================
from threading import Thread
import time
from datetime import datetime

def check_reminders():
    """Vérifie toutes les 1 min les rappels et envoie un email automatique."""
    while True:
        now = datetime.now()
        today = now.strftime("%Y-%m-%d")
        current_time = now.strftime("%H:%M")

        reminders = reminders_collection.find({
            "date": today,
            "time": current_time,
            "notified": False
        })

        for r in reminders:
            user = users_collection.find_one({"username": r["username"]})
            if not user:
                continue

            email = user.get("email") or user.get("username")
            if not email:
                continue

            # 📧 Sujet
            subject = f"📅 Rappel : {r['type']}"

            # 📧 Message HTML avec le template professionnel
            content = email_template(
                f"Rappel : {r['type']}",
                f"""
                <p><strong>Type :</strong> {r['type']}</p>
                <p><strong>Description :</strong> {r['message']}</p>
                <p><strong>Date :</strong> {r['date']} à {r['time']}</p>
                <br>
                <p>Merci de vérifier vos rendez-vous médicaux.</p>
                """
            )

            # 📧 Envoi de l'email
            send_email(email, subject, content)

            # Marquer comme envoyé
            reminders_collection.update_one(
                {"_id": r["_id"]},
                {"$set": {"notified": True}}
            )

        # Attente 1 minute
        time.sleep(60)








# ============================================================
# 🚀 LANCEMENT DE L’APPLICATION
# ============================================================
if __name__ == '__main__':
    # Thread pour activer l'envoi automatique d'emails
    Thread(target=check_reminders, daemon=True).start()
    app.run(debug=True)


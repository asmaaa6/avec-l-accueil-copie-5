import re
import unicodedata
import os
from datetime import datetime
from functools import wraps
from flask import Flask, flash, redirect, render_template, request, session, url_for
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor

# ── OCR & PDF ──────────────────────────────────────────────────────────────
import fitz
import pytesseract
from PIL import Image
import io
import shutil

tesseract_path = shutil.which("tesseract")
if not tesseract_path:
    for path in ['/opt/homebrew/bin/tesseract', '/usr/local/bin/tesseract', '/usr/bin/tesseract']:
        if os.path.exists(path):
            tesseract_path = path
            break
if tesseract_path:
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
    print(f"✅ Tesseract trouvé : {tesseract_path}")
else:
    print("⚠️ Tesseract introuvable — OCR désactivé.")

# ── APPLICATION CONFIGURATION ──────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "recrutai-ultra-secret-2026")

# Dossier d'upload pour stocker les avatars des recruteurs
UPLOAD_AVATAR_FOLDER = os.path.join('static', 'uploads', 'avatars')
os.makedirs(UPLOAD_AVATAR_FOLDER, exist_ok=True)

# ── DATABASE CONFIGURATION (PURE SQL) ──────────────────────────────────────
DB_HOST     = os.environ.get("DB_HOST",     "localhost")
DB_NAME     = os.environ.get("DB_NAME",     "yassir_rh")
DB_USER     = os.environ.get("DB_USER",     "asma")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "")
DB_PORT     = os.environ.get("DB_PORT",     "5432")

def get_db_connection():
    # Connexion native PostgreSQL via psycopg2
    conn = psycopg2.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        port=DB_PORT
    )
    return conn

def init_db():
    
    conn = get_db_connection()
    cur = conn.cursor()
    
    # Création des tables si elles n'existent pas
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id SERIAL PRIMARY KEY,
        nom VARCHAR(100) NOT NULL,
        entreprise VARCHAR(150),
        email VARCHAR(100) UNIQUE NOT NULL,
        password VARCHAR(255) NOT NULL,
        avatar VARCHAR(255),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    
    cur.execute("""
    CREATE TABLE IF NOT EXISTS offres (
        id SERIAL PRIMARY KEY,
        titre VARCHAR(255) NOT NULL,
        description TEXT,
        competences TEXT NOT NULL,
        experience VARCHAR(100),
        formation VARCHAR(150),
        user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    
    cur.execute("""
    CREATE TABLE IF NOT EXISTS cvs (
        id SERIAL PRIMARY KEY,
        nom_fichier VARCHAR(255) NOT NULL,
        contenu TEXT,
        score SMALLINT CHECK (score >= 0 AND score <= 100),
        offre_id INTEGER REFERENCES offres(id) ON DELETE CASCADE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    
    # Création de l'utilisateur de démonstration prototype si absent
    cur.execute("SELECT id FROM users WHERE email = %s", ("recruteur@ummto.dz",))
    if not cur.fetchone():
        hashed_pwd = generate_password_hash("tizi2026")
        cur.execute(
            "INSERT INTO users (nom, entreprise, email, password) VALUES (%s, %s, %s, %s)",
            ("Recruteur UMMTO", "UMMTO", "recruteur@ummto.dz", hashed_pwd)
        )
    
    conn.commit()
    cur.close()
    conn.close()
    print("✅ Base de données initialisée avec succès.")

# Initialisation au démarrage
try:
    init_db()
except Exception as e:
    print(f"⚠️ Erreur d'initialisation de la base de données : {e}")

# ── LOGIN DECORATOR ────────────────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs) :
        if not session.get('authed'):
            flash("Veuillez vous connecter pour accéder à RecrutAI.", "error")
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# ── NLP & TEXT PROCESSING ──────────────────────────────────────────────────
def clean_text(text):
    if not text: return ""
    text = text.lower()
    text = "".join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
    text = re.sub(r'[^a-z0-9\s@\.]', ' ', text)
    return " ".join(text.split())

def extract_text_from_pdf(stream_bytes):
    text = ""
    try:
        doc = fitz.open(stream=stream_bytes, filetype="pdf")
        for page in doc:
            text += page.get_text()
        doc.close()
    except Exception:
        pass
    
    # Fallback OCR si le texte extrait est trop court
    if len(clean_text(text)) < 50 and tesseract_path:
        try:
            text = ""
            doc = fitz.open(stream=stream_bytes, filetype="pdf")
            for page in doc:
                pix = page.get_pixmap(dpi=150)
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))
                text += pytesseract.image_to_string(img, lang="fra+eng")
            doc.close()
        except Exception:
            pass
    return text

# ── KNN RANK BOOSTING ALGORITHM ───────────────────────────────────────────
def compute_knn_rank(scores_arr):
    if len(scores_arr) < 2:
        return np.zeros_like(scores_arr)
    # Plus le score est élevé, plus sa distance à la perfection (1.0) est petite
    distances = np.abs(1.0 - scores_arr)
    max_d = np.max(distances) if np.max(distances) > 0 else 1.0
    boost = 1.0 - (distances / max_d)
    return boost

# ── ROUTES ─────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT * FROM users WHERE email = %s", (email,))
        user = cur.fetchone()
        cur.close()
        conn.close()
        
        if user and check_password_hash(user['password'], password):
            session['authed'] = True
            session['user_id'] = user['id']
            session['nom'] = user['nom']
            session['email'] = user['email']
            flash(f"Ravi de vous revoir, {user['nom']} !", "success")
            return redirect(url_for('dashboard'))
        else:
            flash("Identifiants incorrects. Veuillez réessayer.", "error")
            
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        # 1. Récupération des données du formulaire multi-étapes style Instagram
        nom = request.form.get('nom', '').strip()
        entreprise = request.form.get('entreprise', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        avatar_file = request.files.get('avatar')
        
        if not nom or not email or not password:
            flash("Tous les champs obligatoires (Nom, Email, Mot de passe) doivent être remplis.", "error")
            return render_template('register.html')
            
        # 2. Gestion du téléchargement de la photo de profil (Avatar)
        avatar_filename = None
        if avatar_file and avatar_file.filename != '':
            ext = os.path.splitext(avatar_file.filename)[1].lower()
            if ext in ['.png', '.jpg', '.jpeg', '.gif']:
                # Génération d'un nom unique avec un timestamp pour éviter les doublons de fichiers
                avatar_filename = f"avatar_{int(datetime.now().timestamp())}{ext}"
                upload_dir = os.path.join(app.root_path, 'static', 'uploads', 'avatars')
                os.makedirs(upload_dir, exist_ok=True)
                avatar_file.save(os.path.join(upload_dir, avatar_filename))

        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        try:
            hashed_pwd = generate_password_hash(password)
            
            # 3. Insertion SQL native avec l'entreprise et l'avatar, puis retour de l'ID généré
            cur.execute(
                """INSERT INTO users (nom, entreprise, email, password, avatar) 
                   VALUES (%s, %s, %s, %s, %s) RETURNING id;""",
                (nom, entreprise, email, hashed_pwd, avatar_filename)
            )
            new_user = cur.fetchone()
            conn.commit()
            
            # 4. Connexion automatique immédiate (Enregistrement dans la session Flask)
            session['authed'] = True
            session['user_id'] = new_user['id']
            session['nom'] = nom
            session['email'] = email
            
            flash(f"Bienvenue sur RecrutAI, {nom} ! Votre espace a été configuré avec succès.", "success")
            return redirect(url_for('dashboard'))
            
        except psycopg2.IntegrityError:
            conn.rollback()
            flash("Cet email professionnel est déjà enregistré.", "error")
        except Exception as e:
            conn.rollback()
            flash(f"Erreur lors de l'onboarding : {str(e)}", "error")
        finally:
            cur.close()
            conn.close()
            
    return render_template('register.html')
@app.route('/dashboard')
@login_required
def dashboard():
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    # Récupérer les offres créées par l'utilisateur
    cur.execute("SELECT * FROM offres WHERE user_id = %s ORDER BY created_at DESC", (session['user_id'],))
    offres = cur.fetchall()
    
    # Statistiques rapides globales
    cur.execute("SELECT COUNT(*) as count FROM offres WHERE user_id = %s", (session['user_id'],))
    total_offres = cur.fetchone()['count']
    
    cur.execute("""
        SELECT COUNT(*) as count FROM cvs 
        WHERE offre_id IN (SELECT id FROM offres WHERE user_id = %s)
    """, (session['user_id'],))
    total_cvs = cur.fetchone()['count']
    
    cur.close()
    conn.close()
    
    return render_template('dashboard.html', offres=offres, total_offres=total_offres, total_cvs=total_cvs)

@app.route('/analyse')
@login_required
def analyse():
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("SELECT * FROM offres WHERE user_id = %s ORDER BY created_at DESC", (session['user_id'],))
    offres = cur.fetchall()
    cur.close()
    conn.close()
    return render_template('analyse.html', offres=offres)

@app.route('/creer-offre', methods=['POST'])
@login_required
def creer_offre():
    titre = request.form.get('titre', '').strip()
    competences = request.form.get('competences', '').strip()
    description = request.form.get('description', '').strip()
    experience = request.form.get('experience', '').strip()
    formation = request.form.get('formation', '').strip()
    
    if not titre or not competences:
        flash("Le titre et les compétences clés sont obligatoires.", "error")
        return redirect(url_for('analyse'))
        
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO offres (titre, description, competences, experience, formation, user_id) VALUES (%s, %s, %s, %s, %s, %s)",
        (titre, description, competences, experience, formation, session['user_id'])
    )
    conn.commit()
    cur.close()
    conn.close()
    
    flash("Nouvelle offre d'emploi ajoutée avec succès !", "success")
    return redirect(url_for('analyse'))

@app.route('/match', methods=['POST'])
@login_required
def match():
    offre_id = request.form.get('offre_id')
    files = request.files.getlist('cv_files')
    
    if not offre_id or not files or files[0].filename == '':
        flash("Veuillez sélectionner une offre et ajouter au moins un fichier PDF.", "error")
        return redirect(url_for('analyse'))
        
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    # Récupérer l'offre choisie
    cur.execute("SELECT * FROM offres WHERE id = %s AND user_id = %s", (offre_id, session['user_id']))
    offer = cur.fetchone()
    
    if not offer:
        cur.close()
        conn.close()
        flash("Offre introuvable.", "error")
        return redirect(url_for('analyse'))
        
    try:
        keywords = [clean_text(k) for k in offer['competences'].split(',') if clean_text(k)]
        candidats = []
        
        for f in files:
            if not f.filename.lower().endswith('.pdf'):
                continue
            
            stream = f.read()
            raw_text = extract_text_from_pdf(stream)
            cleaned = clean_text(raw_text)
            
            if not cleaned:
                continue
                
            # Calcul du score TF-IDF simplifié (Fréquence des mots clés cibles)
            match_count = 0
            for kw in keywords:
                if kw in cleaned:
                    match_count += 1
            
            base_score = int((match_count / len(keywords) * 100)) if keywords else 0
            
            candidats.append({
                "fichier": secure_filename(f.filename),
                "texte_brut": raw_text,
                "global_score": base_score,
                "niveau": "À évaluer"
            })
            
        if not candidats:
            cur.close()
            conn.close()
            flash("Aucun texte n'a pu être extrait des fichiers PDF fournis.", "error")
            return redirect(url_for('analyse'))
            
        # Algorithme de Boosting KNN si au moins deux profils sont comparés
        if len(candidats) >= 2:
            raw_scores = np.array([c["global_score"] / 100.0 for c in candidats])
            boost_factors = compute_knn_rank(raw_scores)
            for i, c in enumerate(candidats):
                # Hybridation : 80% score de mots clés + 20% boost de proximité KNN
                final_score = min(100, max(0, int((c["global_score"]/100 * 0.8 + boost_factors[i] * 0.2) * 100)))
                c["global_score"] = final_score

        candidats.sort(key=lambda x: x["global_score"], reverse=True)
        
        # Enregistrement des résultats en SQL pur et attribution des labels de niveau
        for rank, c in enumerate(candidats, 1):
            c["rang"] = rank
            s = c["global_score"]
            if s >= 75:   c["niveau"] = "Excellent"
            elif s >= 50: c["niveau"] = "Bon profil"
            elif s >= 30: c["niveau"] = "Partiel"
            else:         c["niveau"] = "Insuffisant"
            
            cur.execute(
                "INSERT INTO cvs (nom_fichier, contenu, score, offre_id) VALUES (%s, %s, %s, %s)",
                (c["fichier"], c["texte_brut"], c["global_score"], offre_id)
            )
            
        conn.commit()
        return render_template("resultats.html", offer=offer, candidats=candidats)
        
    except Exception as e:
        conn.rollback()
        flash(f"Erreur lors du matching : {str(e)}", "error")
        return redirect(url_for('analyse'))
    finally:
        cur.close()
        conn.close()

@app.route('/logout')
def logout():
    session.clear()
    flash("Vous avez été déconnecté avec succès de RecrutAI.", "success")
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True, port=5000)
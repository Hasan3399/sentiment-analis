import io
import json
import math
import os
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd
from flask import Flask, abort, flash, redirect, render_template, request, send_file, url_for
from flask_login import LoginManager, UserMixin, current_user, login_required, login_user, logout_user
from flask_sqlalchemy import SQLAlchemy
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except Exception:
    torch = None
    TRANSFORMERS_AVAILABLE = False

try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except Exception:
    WORDCLOUD_AVAILABLE = False

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / os.getenv("MODEL_DIR", "model_indobert_sentiment")
INSTANCE_DIR = BASE_DIR / "instance"
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
CSV_DIR = UPLOAD_DIR / "csv"
EXPORT_DIR = UPLOAD_DIR / "exports"
WORDCLOUD_DIR = UPLOAD_DIR / "wordcloud"

for directory in [INSTANCE_DIR, UPLOAD_DIR, CSV_DIR, EXPORT_DIR, WORDCLOUD_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "sentiment-dashboard-local-secret")
app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv(
    "DATABASE_URL",
    f"sqlite:///{INSTANCE_DIR / 'sentiment.db'}",
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024
app.config["UPLOAD_FOLDER"] = str(CSV_DIR)

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"
login_manager.login_message_category = "warning"

DEVICE = torch.device("cuda" if torch and torch.cuda.is_available() else "cpu") if torch else "cpu"

if torch:
    torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS", "1")))

MAX_TEXT_LENGTH = int(os.getenv("MAX_TEXT_LENGTH", "64"))
MAX_ANALYSIS_ROWS = int(os.getenv("MAX_ANALYSIS_ROWS", "1000"))
MAX_CACHE_SIZE = int(os.getenv("MAX_CACHE_SIZE", "1000"))
ALLOWED_EXTENSIONS = {"csv", "xlsx", "xls"}

POSITIVE_WORDS = {
    "bagus", "mantap", "puas", "senang", "suka", "terima", "kasih", "cepat",
    "ramah", "baik", "recommended", "keren", "nyaman", "lancar", "mudah",
    "sempurna", "hebat", "cocok", "bermanfaat", "memuaskan", "sukses", "aman",
    "mantul", "top", "kerenn", "berkualitas", "terbaik", "positif", "love",
    "wow", "senyum", "seneng", "bagusss", "mantul",
}

NEGATIVE_WORDS = {
    "buruk", "kecewa", "marah", "jelek", "lambat", "rusak", "salah", "gagal",
    "kesal", "sakit", "mahal", "parah", "mengecewakan", "kurang", "benci",
    "cacat", "komplain", "masalah", "error", "macet", "hilang", "terlambat",
    "tidak", "negatif", "lemah", "kotor", "lama", "lambat", "jelek",
}

STOPWORDS = {
    "dan", "atau", "yang", "dengan", "untuk", "dari", "pada", "ini", "itu",
    "adalah", "adalah", "saya", "kami", "anda", "mereka", "dia", "ke", "di",
    "sebagai", "dalam", "tidak", "akan", "sudah", "telah", "oleh", "per",
    "the", "and", "or", "is", "are", "to", "of", "in", "for", "a", "an",
}

LABEL_MAPPING = {}
MODEL = None
TOKENIZER = None
MODEL_READY = False
MODEL_ERROR = None
MODEL_LOAD_ATTEMPTED = False
MODEL_DOWNLOAD_ATTEMPTED = False
PREDICTION_CACHE = {}


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(30), default="admin")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class DatasetUpload(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    original_filename = db.Column(db.String(255), nullable=False)
    row_count = db.Column(db.Integer, default=0)
    text_column = db.Column(db.String(120), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    results = db.relationship("AnalysisResult", backref="dataset", lazy=True)


class AnalysisResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    dataset_id = db.Column(db.Integer, db.ForeignKey("dataset_upload.id"), nullable=True)
    text = db.Column(db.Text, nullable=False)
    sentiment = db.Column(db.String(50), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    mode = db.Column(db.String(50), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)


class Keyword(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    word = db.Column(db.String(120), nullable=False)
    sentiment = db.Column(db.String(50), nullable=False)
    count = db.Column(db.Integer, default=1)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


def normalize_label(label):
    text = str(label).strip().replace("_", " ").lower()
    if "posit" in text or text in {"positive", "pos"}:
        return "Positif"
    if "neutr" in text or text in {"neutral", "net", "biasa"}:
        return "Netral"
    if "negat" in text or text in {"negative", "neg"}:
        return "Negatif"
    return str(label).strip().title()


def load_label_mapping():
    global LABEL_MAPPING
    path = MODEL_DIR / "label_mapping.json"
    if not path.exists() or path.stat().st_size == 0:
        return
    try:
        with path.open("r", encoding="utf-8") as file:
            LABEL_MAPPING = json.load(file)
    except Exception:
        LABEL_MAPPING = {}


def resolve_label_mapping(label_id):
    label_id_str = str(label_id)
    for key, value in LABEL_MAPPING.items():
        if str(key) == label_id_str:
            return normalize_label(value)
        if str(value) == label_id_str:
            return normalize_label(key)
    return None


def resolve_label(label_id):
    label_id_int = int(label_id)
    label_id_str = str(label_id)
    if hasattr(MODEL, "config") and MODEL.config:
        id2label = getattr(MODEL.config, "id2label", None) or {}
        raw_label = None
        if label_id_int in id2label:
            raw_label = id2label[label_id_int]
        elif label_id_str in id2label:
            raw_label = id2label[label_id_str]
        if raw_label is not None:
            if str(raw_label).upper().startswith("LABEL_"):
                mapped = resolve_label_mapping(label_id_str)
                if mapped:
                    return mapped
            return normalize_label(raw_label)
    mapped = resolve_label_mapping(label_id_str)
    if mapped:
        return mapped
    return f"Kelas {label_id_int}"


def ensure_model_files():
    global MODEL_DOWNLOAD_ATTEMPTED, MODEL_ERROR
    if (MODEL_DIR / "model.safetensors").exists():
        return True
    model_repo_id = os.getenv("HF_MODEL_REPO_ID", "").strip()
    if not model_repo_id:
        MODEL_ERROR = f"File model tidak ditemukan dan HF_MODEL_REPO_ID belum diatur. Folder model: {MODEL_DIR}"
        return False
    if MODEL_DOWNLOAD_ATTEMPTED:
        return False
    MODEL_DOWNLOAD_ATTEMPTED = True
    try:
        from huggingface_hub import snapshot_download
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=model_repo_id,
            token=os.getenv("HF_TOKEN") or None,
            local_dir=str(MODEL_DIR),
            allow_patterns=[
                "*.json",
                "*.txt",
                "*.safetensors",
                "tokenizer.json",
                "tokenizer_config.json",
                "config.json",
                "label_mapping.json",
            ],
        )
        return (MODEL_DIR / "model.safetensors").exists()
    except Exception as exc:
        MODEL_ERROR = f"Gagal mendownload model dari HuggingFace: {type(exc).__name__}: {str(exc)[:500]}"
        return False


def ensure_model_loaded():
    global MODEL, TOKENIZER, MODEL_READY, MODEL_ERROR, MODEL_LOAD_ATTEMPTED
    if MODEL_LOAD_ATTEMPTED:
        return
    MODEL_LOAD_ATTEMPTED = True
    load_label_mapping()
    if not TRANSFORMERS_AVAILABLE:
        MODEL_ERROR = "Library transformers atau torch belum terinstall."
        return
    if not ensure_model_files():
        return
    try:
        TOKENIZER = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
        MODEL = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR), local_files_only=True)
        MODEL.to(DEVICE)
        MODEL.eval()
        MODEL_READY = True
        MODEL_ERROR = None
    except Exception as exc:
        MODEL_READY = False
        MODEL_ERROR = f"{type(exc).__name__}: {str(exc)[:500]}"


def tokenize_text(text):
    return TOKENIZER(
        text,
        truncation=True,
        padding=True,
        max_length=MAX_TEXT_LENGTH,
        return_tensors="pt",
    )


def predict_with_model(text):
    inputs = tokenize_text(text)
    inputs = {key: value.to(DEVICE) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = MODEL(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)
        top_probability, _ = torch.max(probabilities, dim=1)
    label_probs = []
    for index, probability in enumerate(probabilities[0].cpu().tolist()):
        label_probs.append({
            "label": resolve_label(index),
            "confidence": round(float(probability) * 100, 2),
        })
    label_probs.sort(key=lambda item: item["confidence"], reverse=True)
    return {
        "label": label_probs[0]["label"],
        "confidence": round(float(top_probability.item()) * 100, 2),
        "mode": "IndoBERT",
        "label_probs": label_probs,
    }


def predict_with_lexicon(text):
    words = re.findall(r"[a-z0-9]+", text.lower())
    positive_score = sum(1 for word in words if word in POSITIVE_WORDS)
    negative_score = sum(1 for word in words if word in NEGATIVE_WORDS)
    if positive_score > negative_score:
        label = "Positif"
    elif negative_score > positive_score:
        label = "Negatif"
    else:
        label = "Netral"
    confidence = round(50 + min(abs(positive_score - negative_score) * 12, 42), 2)
    label_probs = [
        {"label": label, "confidence": confidence},
        {"label": "Netral", "confidence": round(max(0, 100 - confidence) * 0.65, 2)},
        {"label": "Negatif", "confidence": round(max(0, 100 - confidence) * 0.35, 2)},
    ]
    label_probs.sort(key=lambda item: item["confidence"], reverse=True)
    return {
        "label": label,
        "confidence": confidence,
        "mode": "Demo Lexicon",
        "label_probs": label_probs,
    }


def predict_uncached(text):
    ensure_model_loaded()
    if MODEL_READY and TOKENIZER and MODEL:
        try:
            return predict_with_model(text)
        except Exception as exc:
            return {
                "label": "Netral",
                "confidence": 50.0,
                "mode": "Error Fallback",
                "label_probs": [
                    {"label": "Netral", "confidence": 50.0},
                    {"label": "Positif", "confidence": 25.0},
                    {"label": "Negatif", "confidence": 25.0},
                ],
                "error": str(exc)[:300],
            }
    return predict_with_lexicon(text)


def predict_sentiment(text):
    cache_key = text[:500].lower().strip()
    if cache_key in PREDICTION_CACHE:
        return dict(PREDICTION_CACHE[cache_key])
    result = predict_uncached(text)
    PREDICTION_CACHE[cache_key] = result
    if len(PREDICTION_CACHE) > MAX_CACHE_SIZE:
        PREDICTION_CACHE.pop(next(iter(PREDICTION_CACHE)), None)
    return result


def count_stats(results):
    stats = {"total": len(results), "positif": 0, "netral": 0, "negatif": 0}
    for result in results:
        label = normalize_label(result.sentiment if hasattr(result, "sentiment") else result.get("sentiment", "Netral"))
        if label == "Positif":
            stats["positif"] += 1
        elif label == "Negatif":
            stats["negatif"] += 1
        else:
            stats["netral"] += 1
    return stats


def extract_words(text):
    return [word for word in re.findall(r"[a-z0-9]+", text.lower()) if word not in STOPWORDS and len(word) > 2]


def upsert_keywords(texts_with_labels):
    for label, text in texts_with_labels:
        label = normalize_label(label)
        for word in extract_words(text):
            if word in POSITIVE_WORDS or word in NEGATIVE_WORDS:
                keyword = Keyword.query.filter_by(word=word, sentiment=label).first()
                if keyword:
                    keyword.count += 1
                else:
                    db.session.add(Keyword(word=word, sentiment=label, count=1))
    db.session.commit()


def find_text_column(columns):
    candidates = ["komentar", "comment", "comments", "text", "teks", "review", "ulasan", "caption", "tweet", "content", "isi", "kalimat"]
    normalized = {str(column).lower().strip(): column for column in columns}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    return next((column for column in columns if "komen" in str(column).lower() or "teks" in str(column).lower()), None)


def read_csv_file(file):
    raw_data = file.read()
    for encoding in ["utf-8-sig", "utf-8", "latin1"]:
        try:
            return pd.read_csv(io.BytesIO(raw_data), encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(io.BytesIO(raw_data), encoding="utf-8-sig", errors="ignore")


def read_excel_file(file):
    raw_data = file.read()
    return pd.read_excel(io.BytesIO(raw_data), engine="openpyxl" if file.filename.lower().endswith(".xlsx") else None)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def get_dashboard_stats():
    total = AnalysisResult.query.count()
    positif = AnalysisResult.query.filter_by(sentiment="Positif").count()
    netral = AnalysisResult.query.filter_by(sentiment="Netral").count()
    negatif = AnalysisResult.query.filter_by(sentiment="Negatif").count()
    avg_confidence = db.session.query(db.func.avg(AnalysisResult.confidence)).scalar() or 0
    uploads = DatasetUpload.query.count()
    return {
        "total": total,
        "positif": positif,
        "netral": netral,
        "negatif": negatif,
        "avg_confidence": round(float(avg_confidence), 2),
        "uploads": uploads,
    }


def get_top_keywords(limit=10):
    return Keyword.query.order_by(Keyword.count.desc()).limit(limit).all()


def get_sentiment_score(stats):
    if not stats["total"]:
        return 0
    return round(((stats["positif"] - stats["negatif"]) / stats["total"]) * 100, 2)


def generate_wordcloud():
    if not WORDCLOUD_AVAILABLE:
        return None
    results = AnalysisResult.query.order_by(AnalysisResult.created_at.desc()).limit(300).all()
    text = " ".join([result.text for result in results if result.text.strip()])
    if not text:
        return None
    wordcloud = WordCloud(
        width=1200,
        height=650,
        background_color="white",
        colormap="viridis",
        max_words=120,
        random_state=42,
    ).generate(text)
    filename = f"wordcloud_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    path = WORDCLOUD_DIR / filename
    wordcloud.to_file(str(path))
    return filename


def export_to_excel(results):
    rows = [
        {
            "id": result.id,
            "text": result.text,
            "sentiment": result.sentiment,
            "confidence": result.confidence,
            "mode": result.mode,
            "created_at": result.created_at.strftime("%Y-%m-%d %H:%M:%S"),
        }
        for result in results
    ]
    df = pd.DataFrame(rows)
    filename = f"hasil_sentimen_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    path = EXPORT_DIR / filename
    df.to_excel(path, index=False)
    return path, filename


def export_to_pdf(results, stats):
    filename = f"laporan_sentimen_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    path = EXPORT_DIR / filename
    doc = SimpleDocTemplate(
        str(path),
        pagesize=landscape(A4),
        rightMargin=24,
        leftMargin=24,
        topMargin=24,
        bottomMargin=24,
    )
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph("Laporan Sentiment Analysis Dashboard", styles["Title"]))
    story.append(Spacer(1, 12))
    summary = [
        ["Total", stats["total"]],
        ["Positif", stats["positif"]],
        ["Netral", stats["netral"]],
        ["Negatif", stats["negatif"]],
        ["Sentiment Score", get_sentiment_score(stats)],
    ]
    table = Table(summary, colWidths=[160, 160])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#111827")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 11),
    ]))
    story.append(table)
    story.append(Spacer(1, 18))
    wordcloud_filename = generate_wordcloud() if WORDCLOUD_AVAILABLE and AnalysisResult.query.count() else None
    wordcloud_path = WORDCLOUD_DIR / wordcloud_filename if wordcloud_filename else None
    if wordcloud_path and wordcloud_path.exists():
        story.append(Image(str(wordcloud_path), width=420, height=230))
        story.append(Spacer(1, 18))
    story.append(Paragraph("Top 50 Hasil Analisis", styles["Heading2"]))
    data = [["No", "Sentimen", "Confidence", "Mode", "Text"]]
    for index, result in enumerate(results[:50], 1):
        data.append([
            index,
            result.sentiment,
            f"{result.confidence}%",
            result.mode,
            result.text[:160],
        ])
    pdf_table = Table(data, colWidths=[40, 90, 90, 120, 420])
    pdf_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#111827")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(pdf_table)
    doc.build(story)
    return path, filename


def create_default_admin():
    username = os.getenv("ADMIN_USERNAME", "admin")
    if User.query.filter_by(username=username).first():
        return
    admin = User(
        username=username,
        email=os.getenv("ADMIN_EMAIL", f"{username}@localhost.local"),
        password_hash=generate_password_hash(os.getenv("ADMIN_PASSWORD", "admin123")),
        role="admin",
    )
    db.session.add(admin)
    db.session.commit()


@app.route("/")
def index():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            flash("Login berhasil.", "success")
            return redirect(url_for("dashboard"))
        flash("Username atau password salah.", "danger")
    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("Anda sudah logout.", "success")
    return redirect(url_for("login"))


@app.route("/reset-data", methods=["POST"])
@login_required
def reset_data():
    AnalysisResult.query.delete()
    DatasetUpload.query.delete()
    Keyword.query.delete()
    db.session.commit()

    for path in WORDCLOUD_DIR.glob("*.png"):
        path.unlink()

    for path in EXPORT_DIR.glob("*"):
        if path.is_file():
            path.unlink()

    flash("Data analisis, keyword, word cloud, dan export berhasil direset.", "success")
    return redirect(url_for("dashboard"))


@app.route("/dashboard")
@login_required
def dashboard():
    stats = get_dashboard_stats()
    recent_results = AnalysisResult.query.order_by(AnalysisResult.created_at.desc()).limit(8).all()
    recent_uploads = DatasetUpload.query.order_by(DatasetUpload.created_at.desc()).limit(5).all()
    return render_template(
        "dashboard.html",
        stats=stats,
        recent_results=recent_results,
        recent_uploads=recent_uploads,
        model_ready=MODEL_READY,
        model_error=MODEL_ERROR,
        active_mode="IndoBERT" if MODEL_READY else "Demo Lexicon",
    )


@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    result = None
    if request.method == "POST":
        text = request.form.get("text", "").strip()
        if not text:
            flash("Masukkan komentar terlebih dahulu.", "warning")
        else:
            prediction = predict_sentiment(text)
            result = AnalysisResult(text=text, sentiment=prediction["label"], confidence=prediction["confidence"], mode=prediction["mode"])
            db.session.add(result)
            upsert_keywords([(prediction["label"], text)])
            db.session.commit()
            flash("Analisis berhasil disimpan.", "success")
            return redirect(url_for("predict_result", result_id=result.id))
    return render_template(
        "predict.html",
        result=result,
        model_ready=MODEL_READY,
        model_error=MODEL_ERROR,
        active_mode="IndoBERT" if MODEL_READY else "Demo Lexicon",
    )


@app.route("/predict/<int:result_id>")
@login_required
def predict_result(result_id):
    result = AnalysisResult.query.get_or_404(result_id)
    return render_template(
        "predict.html",
        result=result,
        model_ready=MODEL_READY,
        model_error=MODEL_ERROR,
        active_mode="IndoBERT" if MODEL_READY else "Demo Lexicon",
    )


@app.route("/dataset", methods=["GET", "POST"])
@login_required
def dataset():
    if request.method == "POST":
        file = request.files.get("file")
        if not file or not file.filename:
            flash("Pilih file CSV atau Excel terlebih dahulu.", "warning")
            return render_template("dataset.html")
        if not allowed_file(file.filename):
            flash("Format file harus CSV, XLSX, atau XLS.", "danger")
            return render_template("dataset.html")
        try:
            if file.filename.lower().endswith(".csv"):
                df = read_csv_file(file)
            else:
                df = read_excel_file(file)
        except Exception as exc:
            flash(f"Gagal membaca file: {exc}", "danger")
            return render_template("dataset.html")
        if df.empty:
            flash("File kosong.", "danger")
            return render_template("dataset.html")
        text_column = find_text_column(df.columns)
        if not text_column:
            flash("File harus memiliki kolom komentar, comment, text, review, atau teks.", "danger")
            return render_template("dataset.html")
        sample_size = min(MAX_ANALYSIS_ROWS, len(df))
        upload = DatasetUpload(
            filename=secure_filename(file.filename),
            original_filename=file.filename,
            row_count=len(df),
            text_column=text_column,
        )
        db.session.add(upload)
        db.session.flush()
        results = []
        for value in df[text_column].head(sample_size).fillna(""):
            text = str(value).strip()
            if not text:
                continue
            prediction = predict_sentiment(text)
            result = AnalysisResult(
                dataset_id=upload.id,
                text=text,
                sentiment=prediction["label"],
                confidence=prediction["confidence"],
                mode=prediction["mode"],
            )
            results.append(result)
            db.session.add(result)
        db.session.commit()
        upsert_keywords([(result.sentiment, result.text) for result in results])
        generate_wordcloud()
        flash(f"Berhasil menganalisis {len(results)} baris dari {sample_size} baris.", "success")
        return redirect(url_for("dataset_result", dataset_id=upload.id))
    uploads = DatasetUpload.query.order_by(DatasetUpload.created_at.desc()).limit(10).all()
    return render_template("dataset.html", uploads=uploads, max_analysis_rows=MAX_ANALYSIS_ROWS)


@app.route("/dataset/<int:dataset_id>")
@login_required
def dataset_result(dataset_id):
    dataset = DatasetUpload.query.get_or_404(dataset_id)
    results = AnalysisResult.query.filter_by(dataset_id=dataset.id).order_by(AnalysisResult.created_at.desc()).limit(200).all()
    stats = count_stats(results)
    avg_confidence = round(sum(result.confidence for result in results) / len(results), 2) if results else 0
    return render_template("dataset_result.html", dataset=dataset, results=results, stats=stats, avg_confidence=avg_confidence)


@app.route("/history")
@login_required
def history():
    page = request.args.get("page", 1, type=int)
    pagination = AnalysisResult.query.order_by(AnalysisResult.created_at.desc()).paginate(page=page, per_page=20, error_out=False)
    return render_template("history.html", pagination=pagination)


@app.route("/visualisasi")
@login_required
def visualisasi():
    if AnalysisResult.query.count():
        generate_wordcloud()
    stats = get_dashboard_stats()
    keywords = get_top_keywords(15)
    wordcloud_files = sorted(WORDCLOUD_DIR.glob("*.png"), key=lambda path: path.stat().st_mtime, reverse=True)
    latest_wordcloud = wordcloud_files[0].name if wordcloud_files else None
    return render_template(
        "visualisasi.html",
        stats=stats,
        keywords=keywords,
        latest_wordcloud=latest_wordcloud,
    )


@app.route("/insight")
@login_required
def insight():
    stats = get_dashboard_stats()
    keywords = get_top_keywords(20)
    positive_keywords = [keyword for keyword in keywords if keyword.sentiment == "Positif"][:10]
    negative_keywords = [keyword for keyword in keywords if keyword.sentiment == "Negatif"][:10]
    avg_confidence = stats["avg_confidence"]
    confidence_level = "Tinggi" if avg_confidence >= 80 else "Sedang" if avg_confidence >= 60 else "Rendah"
    return render_template(
        "insight.html",
        stats=stats,
        positive_keywords=positive_keywords,
        negative_keywords=negative_keywords,
        confidence_level=confidence_level,
        sentiment_score=get_sentiment_score(stats),
    )


@app.route("/export/excel")
@login_required
def export_excel():
    results = AnalysisResult.query.order_by(AnalysisResult.created_at.desc()).limit(5000).all()
    if not results:
        flash("Belum ada data untuk diekspor.", "warning")
        return redirect(url_for("history"))
    path, filename = export_to_excel(results)
    return send_file(path, as_attachment=True, download_name=filename)


@app.route("/export/pdf")
@login_required
def export_pdf():
    results = AnalysisResult.query.order_by(AnalysisResult.created_at.desc()).limit(200).all()
    if not results:
        flash("Belum ada data untuk diekspor.", "warning")
        return redirect(url_for("history"))
    stats = get_dashboard_stats()
    path, filename = export_to_pdf(results, stats)
    return send_file(path, as_attachment=True, download_name=filename)


def init_app():
    with app.app_context():
        db.create_all()
        create_default_admin()


init_app()


if __name__ == "__main__":
    print("APP DIMUAT")
    print(f"DATABASE: {app.config['SQLALCHEMY_DATABASE_URI']}")
    print(f"MODEL READY: {MODEL_READY}")
    print(f"ACTIVE MODE: {'IndoBERT' if MODEL_READY else 'Demo Lexicon'}")
    if not MODEL_READY:
        print(f"MODEL ERROR: {MODEL_ERROR}")
    print("SERVER DIMULAI")
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8080")),
        debug=os.getenv("FLASK_ENV") == "development",
        use_reloader=False,
    )

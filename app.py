
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from routes.routes import register_routes
from routes.rag_core import initialize_rag_system
from routes.config import SECRET_KEY, SQLALCHEMY_DATABASE_URI, SQLALCHEMY_TRACK_MODIFICATIONS

# Route untuk serve file jurnal_ilmiah secara eksplisit (setelah app didefinisikan)
from flask import send_from_directory
import os


# --- Inisialisasi Aplikasi ---
app = Flask(__name__)

# --- Konfigurasi Aplikasi ---
app.config['SECRET_KEY'] = SECRET_KEY
app.config['SQLALCHEMY_DATABASE_URI'] = SQLALCHEMY_DATABASE_URI
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = SQLALCHEMY_TRACK_MODIFICATIONS

# --- Inisialisasi SQLAlchemy ---
from routes.models import db  # Impor db dari routes/models.py
db.init_app(app)



@app.route('/static/jurnal_ilmiah/<path:filename>')
def download_jurnal(filename):
    jurnal_dir = os.path.join(os.path.dirname(__file__), 'jurnal_ilmiah')
    return send_from_directory(jurnal_dir, filename, as_attachment=True)

# --- Registrasi Routes ---
with app.app_context():
    register_routes(app)  # Hanya mengoper app, bukan db
    try:
        db.create_all()
    except Exception as e:
        print("⚠️ Gagal membuat tabel:", e)

    # --- Inisialisasi Sistem RAG ---
    try:
        initialize_rag_system()
        print("✅ RAG system berhasil diinisialisasi.")
    except Exception as e:
        print("⚠️ Gagal inisialisasi RAG system:", e)

# --- Jalankan Aplikasi ---
if __name__ == '__main__':
    app.run(debug=True, port=5000)

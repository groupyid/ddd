import os


# --- Secret Key (harus diganti saat produksi) ---
SECRET_KEY = os.getenv('SECRET_KEY', '7f41b1f8f1a9e0f8c9b01d985e64d1d201edb5bbf7c36b876f8bbd4e789d7a20')

# --- Konfigurasi Database ---
SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL', 'postgresql://postgres:Gemastikusk23@localhost:5432/agrolldb')
SQLALCHEMY_TRACK_MODIFICATIONS = False

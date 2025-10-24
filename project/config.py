import os

class Config:
    SECRET_KEY = 'your_secret_key_here'
    UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

    # MySQL Configuration
    DB_HOST = 'localhost'
    DB_USER = 'root'
    DB_PASSWORD = 'Afnan@9148168146'
    DB_NAME = 'handwriting_gender_classification'

    # Server Settings
    HOST = '0.0.0.0'
    PORT = 5000
    DEBUG = True

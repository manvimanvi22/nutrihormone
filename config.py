import os
from datetime import timedelta

class Config:
    """Base configuration"""
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    UPLOAD_FOLDER = 'uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    PERMANENT_SESSION_LIFETIME = timedelta(days=7)
    
    # Version tracking
    EXTRACTION_METHOD_VERSION = os.getenv('EXTRACTION_METHOD_VERSION', '1.2')
    ML_MODEL_VERSION = os.getenv('ML_MODEL_VERSION', '1.0')
    GEMINI_PROMPT_VERSION = os.getenv('GEMINI_PROMPT_VERSION', '1.0')
    
    # Feature flags
    ENABLE_NEW_PIPELINE = os.getenv('ENABLE_NEW_PIPELINE', 'False').lower() == 'true'
    ENABLE_NEW_CYCLE = os.getenv('ENABLE_NEW_CYCLE', 'False').lower() == 'true'
    ENABLE_DOCTOR_FLAGS = os.getenv('ENABLE_DOCTOR_FLAGS', 'True').lower() == 'true'

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = os.getenv(
        'DATABASE_URL',
        'sqlite:///nutrihormone.db'
    )

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    SQLALCHEMY_DATABASE_URI = os.getenv(
        'DATABASE_URL',
        'sqlite:///nutrihormone.db'
    ).replace('postgres://', 'postgresql://')

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'

config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}
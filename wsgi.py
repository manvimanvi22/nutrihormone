"""WSGI entry point for production deployment"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from app import app, db

if __name__ == "__main__":
    app.run()
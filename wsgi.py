import os
from dotenv import load_dotenv
load_dotenv()
from app import app, db

# Auto-create all database tables on startup
with app.app_context():
    db.create_all()
    print("✓ Database tables created/verified")

if __name__ == "__main__":
    app.run()
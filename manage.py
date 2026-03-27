#!/usr/bin/env python
"""Management script for database migrations and development tasks"""
import os
from flask_script import Manager
from flask_migrate import Migrate, MigrateCommand
from app1 import app, db

# Load environment
os.environ.setdefault('FLASK_ENV', 'development')

# Initialize migration
migrate = Migrate(app, db)
manager = Manager(app)

# Add migration command
manager.add_command('db', MigrateCommand)

@manager.command
def create_admin():
    """Create initial admin user"""
    from app1 import User
    admin = User(
        name='Admin',
        email='admin@nutrihormone.com',
        password='hashed_password_here'
    )
    db.session.add(admin)
    db.session.commit()
    print("Admin user created")

@manager.command
def init_db():
    """Initialize database"""
    db.create_all()
    print("Database initialized")

@manager.command
def drop_db():
    """Drop all tables"""
    if input("Are you sure? Type 'yes': ") == 'yes':
        db.drop_all()
        print("Database dropped")

if __name__ == '__main__':
    manager.run()

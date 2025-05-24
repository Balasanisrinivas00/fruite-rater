"""
Authentication routes for the Fruit Quality Classifier application
"""
from flask import Blueprint, request, jsonify, session
from werkzeug.security import generate_password_hash, check_password_hash
import functools
import os
import json

# Create blueprint
auth_bp = Blueprint('auth', __name__)

# Path to users file
USERS_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'users.json')

def init_users_file():
    """Initialize users file if it doesn't exist"""
    os.makedirs(os.path.dirname(USERS_FILE), exist_ok=True)
    if not os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'w') as f:
            json.dump({}, f)

def get_users():
    """Get all users from the users file"""
    init_users_file()
    try:
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    except:
        return {}

def save_users(users):
    """Save users to the users file"""
    init_users_file()
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f)

def login_required(view):
    """Decorator to require login for routes"""
    @functools.wraps(view)
    def wrapped_view(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({'error': 'Authentication required'}), 401
        return view(*args, **kwargs)
    return wrapped_view

@auth_bp.route('/register', methods=['POST'])
def register():
    """Register a new user"""
    data = request.get_json()
    
    if not data or 'username' not in data or 'password' not in data:
        return jsonify({'error': 'Username and password are required'}), 400
    
    username = data['username']
    password = data['password']
    
    users = get_users()
    
    if username in users:
        return jsonify({'error': 'Username already exists'}), 400
    
    users[username] = {
        'password': generate_password_hash(password),
        'history': []
    }
    
    save_users(users)
    
    return jsonify({'message': 'User registered successfully'}), 201

@auth_bp.route('/login', methods=['POST'])
def login():
    """Log in a user"""
    data = request.get_json()
    
    if not data or 'username' not in data or 'password' not in data:
        return jsonify({'error': 'Username and password are required'}), 400
    
    username = data['username']
    password = data['password']
    
    users = get_users()
    
    if username not in users or not check_password_hash(users[username]['password'], password):
        return jsonify({'error': 'Invalid username or password'}), 401
    
    session.clear()
    session['user_id'] = username
    
    return jsonify({'message': 'Login successful', 'username': username}), 200

@auth_bp.route('/logout', methods=['POST'])
def logout():
    """Log out a user"""
    session.clear()
    return jsonify({'message': 'Logout successful'}), 200

@auth_bp.route('/user', methods=['GET'])
@login_required
def get_user():
    """Get current user information"""
    username = session.get('user_id')
    users = get_users()
    
    if username not in users:
        session.clear()
        return jsonify({'error': 'User not found'}), 404
    
    return jsonify({
        'username': username,
        'history_count': len(users[username].get('history', []))
    }), 200

@auth_bp.route('/history', methods=['GET'])
@login_required
def get_history():
    """Get user prediction history"""
    username = session.get('user_id')
    users = get_users()
    
    if username not in users:
        session.clear()
        return jsonify({'error': 'User not found'}), 404
    
    history = users[username].get('history', [])
    
    return jsonify({'history': history}), 200

def add_to_history(username, prediction_data):
    """Add a prediction to user history"""
    users = get_users()
    
    if username in users:
        if 'history' not in users[username]:
            users[username]['history'] = []
        
        # Limit history to 20 items
        users[username]['history'] = users[username]['history'][-19:] + [prediction_data]
        save_users(users)

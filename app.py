
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import re
import string
from datetime import datetime
import uuid
import json
import os

app = Flask(__name__)
CORS(app)

# Load models
category_model = joblib.load('models/category_model.pkl')
priority_model = joblib.load('models/priority_model.pkl')
sentiment_model = joblib.load('models/sentiment_model.pkl')

# In-memory ticket storage (in production, use a database)
tickets_db = []

# Load existing dataset
if os.path.exists('data/support_tickets.csv'):
    df_existing = pd.read_csv('data/support_tickets.csv')
    for _, row in df_existing.iterrows():
        tickets_db.append({
            'ticket_id': row['ticket_id'],
            'subject': row['subject'],
            'body': row['body'],
            'category': row['category'],
            'priority': row['priority'],
            'sentiment': row['sentiment'],
            'customer_tier': row['customer_tier'],
            'channel': row['channel'],
            'created_at': row['created_at'],
            'status': 'Resolved' if np.random.random() > 0.3 else 'Open',
            'assigned_to': np.random.choice(['Alice Johnson', 'Bob Smith', 'Carol White', 'David Brown', 'Eve Davis', '']),
            'response_time_hours': np.random.randint(1, 72),
            'resolution_time_hours': np.random.randint(2, 168) if np.random.random() > 0.3 else None
        })

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'tkt-\d+|acc-\d+|inv-\d+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 
                  'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                  'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
                  'ought', 'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
                  'from', 'as', 'into', 'through', 'during', 'before', 'after', 'above',
                  'below', 'between', 'under', 'again', 'further', 'then', 'once', 'here',
                  'there', 'when', 'where', 'why', 'how', 'all', 'each', 'few', 'more',
                  'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
                  'same', 'so', 'than', 'too', 'very', 'just', 'and', 'but', 'if', 'or',
                  'because', 'until', 'while', 'this', 'that', 'these', 'those', 'i', 'me',
                  'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
                  'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her',
                  'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
                  'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'am', 'been'}
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    return ' '.join(tokens)

def get_category_icon(category):
    icons = {
        'Technical Issue': '🔧',
        'Billing & Payments': '💳',
        'Account & Access': '🔐',
        'Feature Request': '✨',
        'General Inquiry': '❓',
        'Service Outage': '⚠️'
    }
    return icons.get(category, '📋')

def get_priority_color(priority):
    colors = {
        'High': '#ef4444',
        'Medium': '#f59e0b',
        'Low': '#10b981'
    }
    return colors.get(priority, '#6b7280')

def get_sentiment_emoji(sentiment):
    emojis = {
        'Very Negative': '😡',
        'Negative': '😞',
        'Neutral': '😐',
        'Positive': '😊'
    }
    return emojis.get(sentiment, '😐')

def predict_ticket(text):
    processed = preprocess_text(text)
    category = category_model.predict([processed])[0]
    priority = priority_model.predict([processed])[0]
    
    # Force positive for happy words
    happy_words = ['love', 'amazing', 'fantastic', 'great', 'best', 'perfect', 'excellent', 'happy', 'thank', 'thanks', 'awesome', 'wonderful', 'good', 'nice', 'beautiful', 'brilliant', 'outstanding']
    if any(word in processed for word in happy_words):
        sentiment = 'Positive'
    else:
        sentiment = sentiment_model.predict([processed])[0]

    # Get confidence scores
    cat_conf = np.max(category_model.decision_function([processed]))
    pri_conf = np.max(priority_model.predict_proba([processed])) if hasattr(priority_model, 'predict_proba') else 0.8
    sent_conf = np.max(sentiment_model.predict_proba([processed])) if hasattr(sentiment_model, 'predict_proba') else 0.8

    return {
        'category': category,
        'priority': priority,
        'sentiment': sentiment,
        'confidence': {
            'category': round(float(cat_conf), 3),
            'priority': round(float(pri_conf), 3),
            'sentiment': round(float(sent_conf), 3)
        }
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()
    subject = data.get('subject', '')
    body = data.get('body', '')
    full_text = subject + ' ' + body

    prediction = predict_ticket(full_text)

    # Create ticket
    ticket_id = f'TKT-{str(uuid.uuid4())[:8].upper()}'
    ticket = {
        'ticket_id': ticket_id,
        'subject': subject,
        'body': body,
        'category': prediction['category'],
        'priority': prediction['priority'],
        'sentiment': prediction['sentiment'],
        'customer_tier': data.get('customer_tier', 'Basic'),
        'channel': data.get('channel', 'Web Portal'),
        'created_at': datetime.now().isoformat(),
        'status': 'Open',
        'assigned_to': '',
        'response_time_hours': None,
        'resolution_time_hours': None,
        'confidence': prediction['confidence']
    }
    tickets_db.append(ticket)

    return jsonify({
        'success': True,
        'ticket': ticket,
        'prediction': prediction
    })

@app.route('/api/tickets', methods=['GET'])
def get_tickets():
    status = request.args.get('status', 'all')
    category = request.args.get('category', 'all')
    priority = request.args.get('priority', 'all')
    filtered = tickets_db
    
    # Fix status filtering - handle URL encoding
    if status != 'all':
        # Handle both "in-progress" and "In Progress"
        status_lookup = {
            'open': 'Open',
            'in-progress': 'In Progress',
            'resolved': 'Resolved'
        }
        target_status = status_lookup.get(status.lower(), status)
        filtered = [t for t in filtered if t['status'] == target_status]
    if category != 'all':
        filtered = [t for t in filtered if t['category'] == category]
    if priority != 'all':
        filtered = [t for t in filtered if t['priority'] == priority]

    # Sort by created_at descending
    filtered = sorted(filtered, key=lambda x: x['created_at'], reverse=True)

    return jsonify({
        'success': True,
        'tickets': filtered,
        'total': len(filtered)
    })

@app.route('/api/ticket/<ticket_id>', methods=['GET'])
def get_ticket(ticket_id):
    ticket = next((t for t in tickets_db if t['ticket_id'] == ticket_id), None)
    if ticket:
        return jsonify({'success': True, 'ticket': ticket})
    return jsonify({'success': False, 'error': 'Ticket not found'}), 404

@app.route('/api/ticket/<ticket_id>/status', methods=['PUT'])
def update_status(ticket_id):
    data = request.get_json()
    new_status = data.get('status')

    for ticket in tickets_db:
        if ticket['ticket_id'] == ticket_id:
            ticket['status'] = new_status
            if new_status == 'Resolved' and not ticket.get('resolution_time_hours'):
                ticket['resolution_time_hours'] = np.random.randint(2, 48)
            return jsonify({'success': True, 'ticket': ticket})

    return jsonify({'success': False, 'error': 'Ticket not found'}), 404

@app.route('/api/ticket/<ticket_id>/assign', methods=['PUT'])
def assign_ticket(ticket_id):
    data = request.get_json()
    assigned_to = data.get('assigned_to', '')

    for ticket in tickets_db:
        if ticket['ticket_id'] == ticket_id:
            ticket['assigned_to'] = assigned_to
            return jsonify({'success': True, 'ticket': ticket})

    return jsonify({'success': False, 'error': 'Ticket not found'}), 404

@app.route('/api/dashboard', methods=['GET'])
def dashboard_stats():
    total = len(tickets_db)
    open_count = sum(1 for t in tickets_db if t['status'] == 'Open')
    resolved_count = sum(1 for t in tickets_db if t['status'] == 'Resolved')
    in_progress = sum(1 for t in tickets_db if t['status'] == 'In Progress')

    category_counts = {}
    for t in tickets_db:
        cat = t['category']
        category_counts[cat] = category_counts.get(cat, 0) + 1

    priority_counts = {}
    for t in tickets_db:
        pri = t['priority']
        priority_counts[pri] = priority_counts.get(pri, 0) + 1

    sentiment_counts = {}
    for t in tickets_db:
        sent = t['sentiment']
        sentiment_counts[sent] = sentiment_counts.get(sent, 0) + 1

    # Avg resolution time
    resolved = [t for t in tickets_db if t.get('resolution_time_hours')]
    avg_resolution = np.mean([t['resolution_time_hours'] for t in resolved]) if resolved else 0

    return jsonify({
        'success': True,
        'stats': {
            'total_tickets': total,
            'open': open_count,
            'resolved': resolved_count,
            'in_progress': in_progress,
            'categories': category_counts,
            'priorities': priority_counts,
            'sentiments': sentiment_counts,
            'avg_resolution_hours': round(avg_resolution, 1)
        }
    })

@app.route('/api/analytics', methods=['GET'])
def analytics():
    # Category distribution over time (monthly)
    df_tickets = pd.DataFrame(tickets_db)
    if len(df_tickets) > 0 and 'created_at' in df_tickets.columns:
        df_tickets['created_at'] = pd.to_datetime(df_tickets['created_at'], errors='coerce')
        df_tickets['month'] = df_tickets['created_at'].dt.strftime('%Y-%m')
        monthly = df_tickets.groupby(['month', 'category']).size().unstack(fill_value=0).to_dict()
    else:
        monthly = {}

    # Agent performance
    agent_stats = {}
    for t in tickets_db:
        agent = t.get('assigned_to', '')
        if agent:
            if agent not in agent_stats:
                agent_stats[agent] = {'total': 0, 'resolved': 0, 'avg_resolution': []}
            agent_stats[agent]['total'] += 1
            if t['status'] == 'Resolved':
                agent_stats[agent]['resolved'] += 1
            if t.get('resolution_time_hours'):
                agent_stats[agent]['avg_resolution'].append(t['resolution_time_hours'])

    for agent in agent_stats:
        times = agent_stats[agent]['avg_resolution']
        agent_stats[agent]['avg_resolution'] = round(np.mean(times), 1) if times else 0

    return jsonify({
        'success': True,
        'monthly_trends': monthly,
        'agent_performance': agent_stats
    })

if __name__ == '__main__':
    import webbrowser
    import threading
    threading.Timer(1.5, lambda: webbrowser.open('http://localhost:5000')).start()
    app.run(debug=True, host='0.0.0.0', port=5000)

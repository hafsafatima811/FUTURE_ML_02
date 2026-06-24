# SupportAI - Intelligent Support Ticket Classification System

> **Future Interns - Machine Learning Track | Task 2**

An ML-powered system that automatically classifies customer support tickets by category, assigns priority levels, and analyzes customer sentiment — all through a modern interactive web dashboard.

---

## Features

| Feature | Description |
|---------|-------------|
| **Text Preprocessing** | Cleaning, tokenization, stopword removal, special ID filtering |
| **Category Classification** | 6 categories: Technical Issue, Billing & Payments, Account & Access, Feature Request, General Inquiry, Service Outage |
| **Priority Tagging** | Automatic High / Medium / Low assignment |
| **Sentiment Analysis** | 4-level detection: Very Negative, Negative, Neutral, Positive |
| **Confidence Scoring** | Visual confidence bars for each prediction |
| **Interactive Dashboard** | Real-time stats, doughnut charts, bar charts, ticket management |
| **Ticket Management** | Create, filter by status, update status, assign agents |
| **Analytics** | Monthly trends, sentiment polar area chart, agent performance |

---

## Tech Stack

**Backend:** Python, Flask, scikit-learn, pandas, numpy, NLTK, joblib  
**Frontend:** HTML5, CSS3, Vanilla JavaScript, Chart.js, Font Awesome  
**ML Models:** LinearSVC + TF-IDF (Category), Logistic Regression + TF-IDF (Priority & Sentiment)

---

## Project Structure

```
FUTURE_ML_02/
├── app.py                  # Flask backend API
├── ml_pipeline.py          # Standalone ML training script
├── requirements.txt        # Python dependencies
├── setup.py                # Automated setup script
├── download_nltk.py        # NLTK data downloader
├── README.md               # This file
├── data/
│   └── support_tickets.csv # 5,000 ticket dataset
├── models/
│   ├── category_model.pkl  # Trained category classifier
│   ├── priority_model.pkl  # Trained priority classifier
│   └── sentiment_model.pkl # Trained sentiment analyzer
├── templates/
│   └── index.html          # Single-page web application
├── .vscode/
│   ├── launch.json         # VS Code debug config
│   └── settings.json       # VS Code settings
├── VS_CODE_SETUP.md        # Detailed VS Code setup guide
└── PROJECT_SUMMARY.md      # Full project summary
```

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download NLTK Data
```bash
python download_nltk.py
```

### 3. Run the Application
```bash
python app.py
```

### 4. Open in Browser
Navigate to `http://localhost:5000`

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main dashboard |
| `/api/predict` | POST | Classify a new ticket |
| `/api/tickets` | GET | List all tickets (with filters) |
| `/api/ticket/<id>` | GET | Get ticket details |
| `/api/ticket/<id>/status` | PUT | Update ticket status |
| `/api/ticket/<id>/assign` | PUT | Assign ticket to agent |
| `/api/dashboard` | GET | Dashboard statistics |
| `/api/analytics` | GET | Analytics data |

---

## Model Performance

| Model | Accuracy | Algorithm |
|-------|----------|-----------|
| Category Classifier | ~99% | LinearSVC + TF-IDF |
| Priority Classifier | ~65% | Logistic Regression + TF-IDF |
| Sentiment Analyzer | ~38% | Logistic Regression + TF-IDF |

---

## Demo

The system includes a fully functional web interface with:
- **Dashboard** — Real-time stats with animated cards, category doughnut chart, priority bar chart
- **Classify Ticket** — AI-powered classification with confidence scores & suggested actions
- **All Tickets** — Filterable list with status badges and quick-view modal
- **Analytics** — Monthly trend line charts & sentiment polar area chart

---

## Future Enhancements

- Deep Learning models (BERT, RoBERTa)
- Real-time email / Slack integration
- Multi-language support
- Advanced SLA tracking
- Automated response generation

---

**Built for Future Interns - Machine Learning Track (Task 2)**

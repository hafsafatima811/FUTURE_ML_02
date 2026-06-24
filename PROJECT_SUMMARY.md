# Support Ticket Classification System - Project Summary

## Task 2: Support Ticket Classification (Future Interns - ML Track)

### Requirements Met

#### Core Requirements (from Task 2 brief)
1. **Text Cleaning & Tokenization** - Implemented comprehensive preprocessing pipeline
   - Lowercasing, URL removal, email removal, punctuation removal
   - Stopword removal, lemmatization-ready tokenization
   - Special ID pattern removal (ticket IDs, account IDs)

2. **Ticket Category Classification** - 6 categories with high accuracy
   - Technical Issue
   - Billing & Payments
   - Account & Access
   - Feature Request
   - General Inquiry
   - Service Outage
   - **Accuracy: 99.9%**

3. **Priority Tagging** - Automatic High/Medium/Low assignment
   - Uses Logistic Regression with balanced class weights
   - Correlates with keywords, sentiment, and urgency phrases
   - **Accuracy: 64.7%**

4. **Model Performance Evaluation** - Full classification reports
   - Precision, Recall, F1-score for all classes
   - Confusion matrix analysis
   - Cross-validation on stratified test sets

#### Extra Features Added
1. **Sentiment Analysis** - 4-level sentiment detection
   - Very Negative, Negative, Neutral, Positive
   - Helps gauge customer frustration level
   - **Accuracy: 38.0%**

2. **Confidence Scoring** - Shows prediction confidence
   - Visual confidence bars for each prediction
   - Helps identify uncertain classifications

3. **Interactive Web Dashboard** - Modern single-page application
   - Real-time statistics with animated cards
   - Doughnut charts (Category distribution)
   - Bar charts (Priority distribution)
   - Line charts (Monthly trends)
   - Polar area charts (Sentiment analysis)

4. **Ticket Management System**
   - Create new tickets with AI classification
   - View all tickets with filtering (status, category, priority)
   - Update ticket status (Open → In Progress → Resolved)
   - Assign tickets to agents
   - Search functionality

5. **AI-Suggested Actions**
   - Based on category + priority + sentiment
   - Recommends next steps for support team
   - Escalation suggestions for high-priority items

6. **Modern UI/UX Design**
   - Dark sidebar with glassmorphism effect
   - Animated floating particles background
   - Responsive design (desktop, tablet, mobile)
   - Smooth transitions and hover effects
   - Toast notifications for user feedback
   - Loading overlays with spinners
   - Modal dialogs for ticket details

7. **Analytics & Reporting**
   - Agent performance metrics
   - Average resolution time tracking
   - Monthly trend analysis
   - Category/priority distribution

8. **Complete Dataset**
   - 5,000 synthetic support tickets
   - Realistic subjects and descriptions
   - Multiple attributes (tier, channel, timestamps)
   - Balanced across all categories

### Tech Stack

**Backend:**
- Python 3.10+
- Flask (Web framework)
- scikit-learn (ML models)
- NLTK (NLP preprocessing)
- joblib (Model serialization)
- pandas, numpy (Data processing)

**Frontend:**
- HTML5 + CSS3 (No framework dependency)
- Vanilla JavaScript
- Chart.js (Data visualization)
- Font Awesome (Icons)
- Google Fonts (Inter)

**ML Models:**
- Category: LinearSVC + TF-IDF (99.9% accuracy)
- Priority: LogisticRegression + TF-IDF (64.7% accuracy)
- Sentiment: LogisticRegression + TF-IDF (38.0% accuracy)

### File Structure
```
support_ticket_classifier/
├── app.py                  # Flask backend API
├── ml_pipeline.py          # Standalone ML training script
├── requirements.txt        # Python dependencies
├── README.md              # Documentation
├── .gitignore             # Git ignore rules
├── templates/
│   └── index.html         # Single-page web app
├── static/                # Static assets
├── models/
│   ├── category_model.pkl # Trained category classifier
│   ├── priority_model.pkl # Trained priority classifier
│   └── sentiment_model.pkl# Trained sentiment analyzer
└── data/
    └── support_tickets.csv # 5000 ticket dataset
```

### How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the web application:
   ```bash
   python app.py
   ```

3. Open browser at `http://localhost:5000`

### API Endpoints
- `POST /api/predict` - Classify new ticket
- `GET /api/tickets` - List all tickets
- `GET /api/ticket/<id>` - Get ticket details
- `PUT /api/ticket/<id>/status` - Update status
- `GET /api/dashboard` - Dashboard stats
- `GET /api/analytics` - Analytics data

### Sample Predictions (Verified)
| Input | Category | Priority | Sentiment |
|-------|----------|----------|-----------|
| "Production server down, 503 errors" | Service Outage | High | Very Negative |
| "Charged twice for subscription" | Billing & Payments | Medium | Negative |
| "Forgot password, reset email not arriving" | Account & Access | High | Very Negative |
| "Add dark mode to dashboard" | Feature Request | Low | Neutral |
| "How to set up webhooks?" | General Inquiry | Low | Neutral |
| "Mobile app crashes on iPhone" | Technical Issue | Medium | Neutral |

### Repository
**GitHub Repository Name:** `FUTURE_ML_02`

# VS Code Setup Guide - Support Ticket Classification System

## Step-by-Step Instructions to Run in VS Code

---

## Step 1: Install VS Code

1. Download VS Code from https://code.visualstudio.com
2. Install it on your system (Windows/Mac/Linux)

---

## Step 2: Install Python

1. Download Python 3.10+ from https://www.python.org/downloads
2. **IMPORTANT**: During installation, check "Add Python to PATH"
3. Verify installation by opening Command Prompt/Terminal and typing:
   ```
   python --version
   ```
   Should show: Python 3.10.x or higher

---

## Step 3: Download the Project

### Option A: Download ZIP
1. Download the project ZIP file
2. Extract it to a folder (e.g., C:\Users\YourName\Projects\FUTURE_ML_02)

### Option B: Clone from GitHub (if you uploaded)
```bash
git clone https://github.com/YOUR_USERNAME/FUTURE_ML_02.git
```

---

## Step 4: Open Project in VS Code

1. Open VS Code
2. Click File -> Open Folder (or Ctrl+K Ctrl+O)
3. Select the extracted FUTURE_ML_02 folder
4. VS Code will open the project

---

## Step 5: Install VS Code Extensions

Click the Extensions icon (left sidebar, looks like 4 squares) or press Ctrl+Shift+X

Install these extensions:
- **Python** (by Microsoft) - REQUIRED
- **Pylance** (by Microsoft) - Auto-installed with Python
- **Jupyter** (by Microsoft) - For notebooks
- **Live Server** (by Ritwick Dey) - For frontend preview

---

## Step 6: Create Virtual Environment (RECOMMENDED)

### Using VS Code Terminal:

1. Open Terminal in VS Code: Terminal -> New Terminal (or Ctrl+`)

2. Create virtual environment:
   ```bash
   # Windows
   python -m venv venv

   # Mac/Linux
   python3 -m venv venv
   ```

3. Activate virtual environment:
   ```bash
   # Windows
   venv\Scripts\activate

   # Mac/Linux
   source venv/bin/activate
   ```

   You'll see (venv) in your terminal prompt when activated

---

## Step 7: Install Dependencies

With virtual environment activated, run:

```bash
pip install -r requirements.txt
```

This installs:
- Flask
- flask-cors
- scikit-learn
- pandas
- numpy
- nltk
- joblib

---

## Step 8: Download NLTK Data

Create a file `download_nltk.py` in VS Code:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
print("NLTK data downloaded successfully!")
```

Run it in terminal:
```bash
python download_nltk.py
```

---

## Step 9: Run the Application

### Method 1: Run Flask Backend

1. Make sure you're in the project root folder (where app.py is)
2. In VS Code terminal:
   ```bash
   python app.py
   ```
3. You should see:
   ```
   * Running on http://127.0.0.1:5000
   * Running on http://192.168.x.x:5000
   ```

4. Open browser and go to: http://localhost:5000

### Method 2: Run ML Pipeline Only (No Web UI)

```bash
# Train models
python ml_pipeline.py --train

# Interactive prediction
python ml_pipeline.py --predict

# Evaluate models
python ml_pipeline.py --evaluate
```

---

## Step 10: VS Code Debug Configuration (Optional)

Create .vscode/launch.json for easy debugging:

1. Press Ctrl+Shift+D (Debug panel)
2. Click "create a launch.json file"
3. Select "Python -> Flask"
4. VS Code will create the configuration

Or manually create .vscode/launch.json:
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Flask",
            "type": "python",
            "request": "launch",
            "module": "flask",
            "env": {
                "FLASK_APP": "app.py",
                "FLASK_DEBUG": "1"
            },
            "args": [
                "run",
                "--no-debugger",
                "--no-reload"
            ],
            "jinja": true,
            "justMyCode": true
        }
    ]
}
```

Now press F5 to debug!

---

## Step 11: VS Code Settings (Recommended)

Create .vscode/settings.json:
```json
{
    "python.defaultInterpreterPath": "./venv/Scripts/python.exe",
    "python.terminal.activateEnvironment": true,
    "editor.formatOnSave": true,
    "python.formatting.provider": "black",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true
}
```

---

## Troubleshooting

### Issue: "python" command not found
**Fix**: Add Python to PATH during installation, or use `py` instead of `python` on Windows

### Issue: "pip" command not found
**Fix**: Use `python -m pip install ...` instead of `pip install ...`

### Issue: Port 5000 already in use
**Fix**: Change port in app.py:
```python
app.run(debug=True, host='0.0.0.0', port=5001)  # Use 5001 instead
```

### Issue: Module not found errors
**Fix**: Make sure virtual environment is activated and requirements are installed

### Issue: NLTK data not found
**Fix**: Run the NLTK download script again (Step 8)

---

## Quick Start Commands (Copy-Paste)

```bash
# 1. Open terminal in VS Code
# Terminal -> New Terminal

# 2. Create and activate venv
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download NLTK data
python download_nltk.py

# 5. Run the app
python app.py

# 6. Open browser to http://localhost:5000
```

---

## File Structure in VS Code

```
FUTURE_ML_02/
├── .vscode/
│   ├── launch.json          <- Debug config
│   └── settings.json        <- VS Code settings
├── venv/                    <- Virtual environment (auto-created)
├── data/
│   └── support_tickets.csv
├── models/
│   ├── category_model.pkl
│   ├── priority_model.pkl
│   └── sentiment_model.pkl
├── templates/
│   └── index.html
├── static/                  <- (empty, for future assets)
├── app.py                   <- Main Flask app
├── ml_pipeline.py           <- Standalone ML script
├── requirements.txt         <- Dependencies
├── README.md               <- Documentation
├── .gitignore              <- Git ignore rules
└── download_nltk.py        <- NLTK data downloader
```

---

## Keyboard Shortcuts in VS Code

| Shortcut | Action |
|----------|--------|
| Ctrl+Shift+P | Command Palette |
| Ctrl+Shift+X | Extensions |
| Ctrl+Shift+D | Debug |
| Ctrl+` | Terminal |
| Ctrl+Shift+O | Open Folder |
| F5 | Start Debugging |
| Ctrl+C (in terminal) | Stop Flask server |

---

Happy Coding!

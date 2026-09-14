# FinancialRAG - Stock Market Prediction Application

AI-powered financial news assistant and stock market prediction platform built with Django.

## Features

- **Home**: AI chat assistant for real-time stock analysis
- **Portfolio**: Track investments with performance charts and AI sentiment analysis
- **News Feed**: Real-time financial news with AI summaries
- **Analytics**: User engagement metrics and data source utilization
- **Settings**: Configure notifications, AI summaries, and news sources

## Requirements

Install all dependencies:

```bash
pip install -r requirements.txt
```

## Running the Application

1. **Activate virtual environment:**
   ```bash
   source venv/bin/activate
   ```

2. **Run migrations:**
   ```bash
   python manage.py migrate
   ```

3. **Start development server:**
   ```bash
   python manage.py runserver
   ```

4. **Access the application:**
   Open http://127.0.0.1:8000/ in your browser

## Project Structure

```
P1/
├── core/                    # Main Django app
│   ├── templates/          # HTML templates (View)
│   │   ├── base.html
│   │   └── core/
│   │       ├── home.html
│   │       ├── portfolio.html
│   │       ├── newsfeed.html
│   │       ├── analytics.html
│   │       └── settings.html
│   ├── static/             # CSS and JavaScript
│   │   ├── css/
│   │   └── js/
│   ├── views.py            # Controllers
│   ├── models.py           # Models
│   └── urls.py             # URL routing
├── p1/                     # Django project settings
├── requirements.txt        # Python dependencies
└── manage.py
```

## Architecture

This project follows the **Model-View-Controller (MVC)** pattern:

- **Model** (`models.py`): Data structures and database models
- **View** (`templates/`): HTML templates with Tailwind CSS
- **Controller** (`views.py`): Business logic and request handling

## Technologies

- **Backend**: Django 6.1.1
- **Frontend**: Tailwind CSS, Chart.js
- **ML Libraries**: NumPy, Pandas, Scikit-learn, PyTorch
- **Data**: yfinance, TA-Lib

# Personal Fitness Tracker

An AI-powered fitness tracker using Python and machine learning for personalized workout plans, calorie burn prediction, and activity monitoring. It integrates with Google Fit API for real-time tracking.

## Key Features
- Personalized Workouts: Custom routines based on user input.  
- Calorie Prediction: Real-time estimation using ML models.  
- Activity Monitoring: Track steps, heart rate, and progress.  
- Interactive Dashboard: Visualize results via Streamlit.  
- Google Fit Integration: Syncs with user fitness data.  

## Technologies Used
- Frontend: Streamlit (UI & Dashboard)  
- Backend: Python (ML models, API handling)  
- ML Models: Random Forest, Logistic Regression, LSTM  
- APIs: Google Fit API, Streamlit Cloud  

## Workflow
1. User Input: Collects data (age, weight, etc.)  
2. ML Prediction: Analyzes inputs for calorie estimates.  
3. Dashboard: Displays results and workout suggestions.  


## Setup Instructions

1. Install Dependencies: 
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt

2. Run the Application:  
   streamlit run app/app.py
   


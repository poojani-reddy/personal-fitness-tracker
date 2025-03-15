import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import time
import warnings
import io
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

warnings.filterwarnings('ignore')

# Title and Description
st.title("Personal Fitness Tracker")

st.write("""
In this WebApp, you can track and estimate the calories you burn during physical activities.  
- Input personal and workout details (Age, BMI, Exercise Duration, Heart Rate, etc.)  
- Predict the calories burned using an advanced Random Forest model  
- Gain insights by comparing your profile with similar workout records  
- Receive personalized feedback on exercise intensity and health metrics

Simply enter your details in the sidebar and get an accurate calorie prediction!
""")

# Sidebar for User Input
st.sidebar.header("User Input Parameters")

def user_input_features():
    age = st.sidebar.slider("Age", 10, 100, 30)
    bmi = st.sidebar.slider("BMI", 15.0, 40.0, 22.5)
    duration = st.sidebar.slider("Exercise Duration (min)", 0, 120, 30)
    heart_rate = st.sidebar.slider("Heart Rate (bpm)", 60, 200, 85)
    body_temp = st.sidebar.slider("Body Temperature (°C)", 35.0, 42.0, 37.0)
    gender = st.sidebar.radio("Gender", ["Male", "Female"])

    gender_encoded = 1 if gender == "Male" else 0

    features = pd.DataFrame({
        'Age': [age],
        'BMI': [bmi],
        'Duration': [duration],
        'Heart_Rate': [heart_rate],
        'Body_Temp': [body_temp],
        'Gender_male': [gender_encoded]
    })
    return features

user_data = user_input_features()

st.write("### Your Input Parameters")
st.dataframe(user_data)

# Load and Preprocess Data
@st.cache_data
def load_data():
    try:
        calories = pd.read_csv("calories.csv")
        exercise = pd.read_csv("exercise.csv")
        return calories, exercise
    except FileNotFoundError as e:
        st.error("Data files not found. Ensure 'calories.csv' and 'exercise.csv' are present.")
        st.stop()

calories, exercise = load_data()

exercise_df = pd.merge(exercise, calories, on='User_ID')
exercise_df.drop(columns='User_ID', inplace=True)

exercise_df['BMI'] = round(exercise_df['Weight'] / ((exercise_df['Height'] / 100) ** 2), 2)

exercise_df = pd.get_dummies(exercise_df, columns=['Gender'], drop_first=True)

# Prepare Training Data
X = exercise_df.drop("Calories", axis=1)
y = exercise_df["Calories"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest Model Training
rf_model = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42)
rf_model.fit(X_train, y_train)

# Align Input with Model Columns
user_data = user_data.reindex(columns=X_train.columns, fill_value=0)

# Make Prediction
prediction = rf_model.predict(user_data)

st.write("### Predicted Calorie Burn")
st.success(f"Estimated Calories Burned: **{round(prediction[0], 2)} kcal**")

# Display Similar Results
st.write("### Similar Workout Profiles")
similar_range = (prediction[0] - 10, prediction[0] + 10)
similar_data = exercise_df[(exercise_df['Calories'] >= similar_range[0]) & (exercise_df['Calories'] <= similar_range[1])]
st.dataframe(similar_data.sample(5) if not similar_data.empty else pd.DataFrame({"Message": ["No similar profiles found"]}))

# Personalized Insights
age_percentile = (user_data['Age'][0] > X_train['Age']).mean() * 100
duration_percentile = (user_data['Duration'][0] > X_train['Duration']).mean() * 100
heart_rate_percentile = (user_data['Heart_Rate'][0] > X_train['Heart_Rate']).mean() * 100
body_temp_percentile = (user_data['Body_Temp'][0] > X_train['Body_Temp']).mean() * 100

st.write("### Personalized Insights")
st.write(f"You are older than **{age_percentile:.2f}%** of other people.")
st.write(f"Your exercise duration is higher than **{duration_percentile:.2f}%** of other people.")
st.write(f"You have a higher heart rate than **{heart_rate_percentile:.2f}%** of other people during exercise.")
st.write(f"You have a higher body temperature than **{body_temp_percentile:.2f}%** of other people during exercise.")

# Actionable Feedback
def generate_feedback(calories):
    if calories < 200:
        return "Consider increasing workout intensity by extending duration or adding higher-effort activities."
    elif calories > 400:
        return "Your calorie burn is high—incorporate rest days to prevent fatigue and promote recovery."
    else:
        return "Your calorie burn is within a healthy range. Maintain your current workout intensity!"

feedback = str(generate_feedback(prediction[0]))
st.write("### Actionable Feedback")
st.write(feedback)

# Visualization
st.write("### Calorie Burn Distribution")
plt.figure(figsize=(8, 5))
sns.kdeplot(exercise_df['Calories'], shade=True, color='skyblue')
plt.axvline(prediction[0], color='red', linestyle='--', label=f"Your Prediction: {round(prediction[0], 2)} kcal")
plt.title("Calories Burned Distribution")
plt.xlabel("Calories Burned")
plt.ylabel("Density")
plt.legend()
st.pyplot(plt)
def create_pdf_report(user_data, prediction, age_percentile, duration_percentile, heart_rate_percentile, body_temp_percentile):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("Personal Fitness Tracker Report", styles['Title']))
    elements.append(Spacer(1, 12))

    for col in user_data.columns:
        elements.append(Paragraph(f"{col}: {user_data[col].values[0]}", styles['Normal']))

    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"Estimated Calories Burned: {round(prediction[0], 2)} kcal", styles['Normal']))

    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"You are older than {age_percentile:.2f}% of other people.", styles['Normal']))
    elements.append(Paragraph(f"Your exercise duration is higher than {duration_percentile:.2f}% of other people.", styles['Normal']))
    elements.append(Paragraph(f"You have a higher heart rate than {heart_rate_percentile:.2f}% of other people during exercise.", styles['Normal']))
    elements.append(Paragraph(f"You have a higher body temperature than {body_temp_percentile:.2f}% of other people during exercise.", styles['Normal']))

    elements.append(Spacer(1, 12))
    elements.append(Paragraph("Actionable Feedback:", styles['Heading2']))
    elements.append(Paragraph(feedback, styles['Normal']))
    doc.build(elements)
    buffer.seek(0)
    return buffer

st.write("### Export Your Report")
if st.button("Download PDF Report"):
    pdf_report = create_pdf_report(user_data, prediction, age_percentile, duration_percentile, heart_rate_percentile, body_temp_percentile)
    st.download_button(label="Download PDF", data=pdf_report, file_name="fitness_report.pdf", mime="application/pdf")
st.write("---")
st.write("© 2025 Personal Fitness Tracker")

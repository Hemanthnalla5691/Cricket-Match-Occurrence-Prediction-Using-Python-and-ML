import pandas as pd
import numpy as np
import gradio as gr
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
data = pd.read_csv(r"C:\Users\LENOVO\Downloads\cricket_matches.csv")
data.dropna(inplace=True)
y = data['match_occurrence']
X = data.drop(columns=['match_occurrence'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
preds = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, preds) * 100:.2f}%")
def predict_match_occurrence(temperature, humidity, wind_speed, cloud_cover):
    features = np.array([[temperature, humidity, wind_speed, cloud_cover]])
    prediction = model.predict(features)[0]
    return "Match Will Occur" if prediction == 1 else "Match Will Be Cancelled"
iface = gr.Interface(
    fn=predict_match_occurrence,
    inputs=[
        gr.Number(label="Temperature (°C)"),
        gr.Number(label="Humidity (%)"),
        gr.Number(label="Wind Speed (km/h)"),
        gr.Number(label="Cloud Cover (%)")
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="Cricket Match Occurrence Predictor",
    description="Enter weather conditions to predict if a cricket match will be held or canceled."
)
iface.launch(debug=True, share=True)
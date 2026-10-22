# 🎓 Teacher Eye – AI-Powered Student Engagement Monitoring

Teacher Eye is an AI-powered student engagement monitoring project designed to detect student behaviors in classroom images using YOLO object detection and provide a simple dashboard through Streamlit.

## 🚀 Features

- 🧠 AI-based student behavior detection using YOLO
- 🖼️ Upload classroom images for prediction
- 📷 Capture images using the camera
- 📊 Streamlit dashboard for displaying prediction results
- 🔐 Login and registration interface
- 📁 Store and view prediction results locally

## 🛠️ Technologies Used

- Python
- YOLO / Ultralytics
- Streamlit
- OpenCV
- Pandas
- Pillow (PIL)

## 📂 Project Structure

```text
TeacherEye-AI-Student-Monitoring/
├── app.py
├── train.py
├── best.pt
├── requirements.txt
├── README.md
└── .gitignore
```

## ⚙️ Setup

1. Clone this repository.
2. Install the required Python packages using `pip install -r requirements.txt`.
3. Ensure the trained YOLO model file is available at the expected path.
4. Start the application using `streamlit run app.py`.

## 🎯 Project Goal

The goal of Teacher Eye is to explore AI-based student engagement monitoring and visualize model predictions to support educators.

## ⚠️ Disclaimer

This project is a prototype for educational purposes. Model predictions may be inaccurate and should not be used as the sole basis for evaluating students. Protect student privacy and obtain appropriate consent before collecting or processing classroom images.

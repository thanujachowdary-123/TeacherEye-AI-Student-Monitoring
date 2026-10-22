
import os
import json
import hashlib
import secrets
from pathlib import Path
from datetime import datetime

import streamlit as st
import pandas as pd
import cv2
from PIL import Image
from ultralytics import YOLO


# ==========================================================
# CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Teacher Eye - AI Student Monitoring",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PREDICTIONS_DIR = DATA_DIR / "predictions"
USERS_FILE = DATA_DIR / "users.json"
RECORDS_FILE = DATA_DIR / "predictions.json"
MODEL_PATH = BASE_DIR / "best.pt"

DATA_DIR.mkdir(exist_ok=True)
PREDICTIONS_DIR.mkdir(exist_ok=True)

# ==========================================================
# CUSTOM DESIGN
# ==========================================================

st.markdown("""
<style>
@import url(
'https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap'
);

.stApp {
    font-family: 'Poppins', sans-serif;
    background-color: #f4f7fc;
}

h1, h2, h3 {
    color: #244c80;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #244c80, #4c8bf5);
}

section[data-testid="stSidebar"] * {
    color: white;
}

div.stButton > button {
    border-radius: 10px;
    border: none;
    padding: 10px 18px;
    font-weight: 600;
}

div.stButton > button:hover {
    border: 1px solid #4c8bf5;
}

.hero {
    background: linear-gradient(120deg, #244c80, #4c8bf5);
    color: white;
    padding: 32px;
    border-radius: 18px;
    text-align: center;
    margin-bottom: 25px;
}

.hero h1 {
    color: white;
    font-size: 42px;
}

.hero p {
    font-size: 17px;
}

.info-card {
    background: white;
    padding: 22px;
    border-radius: 14px;
    border-left: 5px solid #4c8bf5;
    margin-bottom: 16px;
}

</style>
""", unsafe_allow_html=True)


# ==========================================================
# JSON AND USER HELPERS
# ==========================================================

def read_json(path, default):
    if not path.exists():
        return default

    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except (json.JSONDecodeError, OSError):
        return default


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


def hash_password(password, salt=None):
    if salt is None:
        salt = secrets.token_hex(16)

    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        200000
    ).hex()

    return {"salt": salt, "hash": digest}


def verify_password(password, saved):
    if isinstance(saved, str):
        # Supports legacy demo accounts from the original project.
        return secrets.compare_digest(password, saved)

    if not isinstance(saved, dict):
        return False

    salt = saved.get("salt", "")
    expected = saved.get("hash", "")

    if not salt or not expected:
        return False

    actual = hash_password(password, salt)["hash"]
    return secrets.compare_digest(actual, expected)


def load_users():
    return read_json(USERS_FILE, {})


def save_users(users):
    write_json(USERS_FILE, users)


def load_predictions():
    return read_json(RECORDS_FILE, {})


def save_predictions(predictions):
    write_json(RECORDS_FILE, predictions)


def save_latest_image(username, image_path):
    latest = read_json(DATA_DIR / "latest_images.json", {})
    latest[username] = str(image_path)
    write_json(DATA_DIR / "latest_images.json", latest)


def load_latest_image(username):
    latest = read_json(DATA_DIR / "latest_images.json", {})
    return latest.get(username)


# ==========================================================
# MODEL LOADING
# ==========================================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None

    return YOLO(str(MODEL_PATH))


def check_model():
    model = load_model()

    if model is None:
        st.error(
            "Model file best.pt was not found. "
            "Place your trained YOLO model in the project folder."
        )
        return None

    return model


# ==========================================================
# SESSION STATE
# ==========================================================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = None
    st.session_state.role = None
    st.session_state.page = "Home"


def switch_page(page):
    st.session_state.page = page
    st.rerun()


def logout():
    st.session_state.logged_in = False
    st.session_state.username = None
    st.session_state.role = None
    st.session_state.page = "Home"
    st.rerun()


# ==========================================================
# HOME PAGE
# ==========================================================

def home_page():
    st.markdown("""
    <div class="hero">
        <h1>TEACHER EYE</h1>
        <p>AI-Powered Student Engagement Monitoring System</p>
        <p>Computer Vision • YOLOv9 • Python • Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Welcome to Teacher Eye")

    st.write(
        "Teacher Eye is a prototype application that uses "
        "computer vision to detect visible student behaviors "
        "in classroom images."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="info-card">
            <h3>🎯 AI Detection</h3>
            <p>Analyze classroom images using a trained YOLO model.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-card">
            <h3>📊 Teacher Dashboard</h3>
            <p>Review saved detection labels and timestamps.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info-card">
            <h3>📷 Image Analysis</h3>
            <p>Upload an image or capture one using your camera.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-card">
            <h3>🔐 User Accounts</h3>
            <p>Use separate teacher and student application roles.</p>
        </div>
        """, unsafe_allow_html=True)

    st.warning(
        "AI behavior predictions can be incorrect. "
        "Do not use them alone to judge student attention, "
        "performance, or attendance."
    )


# ==========================================================
# LOGIN PAGE
# ==========================================================

def login_page():
    st.title("🔐 Login")

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login", use_container_width=True)

    if submit:
        users = load_users()
        user = users.get(username)

        if user and verify_password(password, user.get("password", "")):
            # Upgrade older plaintext demo passwords after login.
            if isinstance(user.get("password"), str):
                user["password"] = hash_password(password)
                users[username] = user
                save_users(users)

            st.session_state.logged_in = True
            st.session_state.username = username
            st.session_state.role = user.get("role", "Student")

            st.success("Login successful!")

            if st.session_state.role == "Teacher":
                switch_page("StudentActivity")
            else:
                switch_page("Prediction")
        else:
            st.error("Invalid username or password.")


# ==========================================================
# REGISTRATION PAGE
# ==========================================================

def register_page():
    st.title("📝 Create an Account")

    with st.form("register_form"):
        username = st.text_input("Choose a username")
        password = st.text_input(
            "Create a password", type="password"
        )
        confirm_password = st.text_input(
            "Confirm password", type="password"
        )
        role = st.selectbox("Account role", ["Student", "Teacher"])
        submit = st.form_submit_button(
            "Register", use_container_width=True
        )

    if submit:
        username = username.strip()

        if not username or not password:
            st.error("Username and password are required.")

        elif len(password) < 8:
            st.error("Use a password with at least 8 characters.")

        elif password != confirm_password:
            st.error("Passwords do not match.")

        else:
            users = load_users()

            if username in users:
                st.error("This username already exists.")
            else:
                users[username] = {
                    "password": hash_password(password),
                    "role": role
                }

                save_users(users)

                if role == "Student":
                    predictions = load_predictions()
                    predictions[username] = {
                        "engagement_labels": [],
                        "last_updated": "Not yet predicted"
                    }
                    save_predictions(predictions)

                st.success("Account created. Please log in.")
                st.session_state.page = "Login"


# ==========================================================
# STUDENT ACTIVITY PAGE - TEACHER DASHBOARD
# ==========================================================

def student_activity_page():
    if (
        not st.session_state.logged_in
        or st.session_state.role != "Teacher"
    ):
        st.error("Please log in with a teacher account.")
        return

    st.title("📊 Student Activity Dashboard")

    predictions = load_predictions()

    if not predictions:
        st.info("No student records are available yet.")
        return

    rows = []

    for username, info in predictions.items():
        rows.append({
            "Username": username,
            "Detected Behaviors": ", ".join(
                info.get("engagement_labels", [])
            ) or "No detections",
            "Last Updated": info.get(
                "last_updated", "Not yet predicted"
            )
        })

    df = pd.DataFrame(rows)

    col1, col2 = st.columns(2)

    col1.metric("Registered Student Records", len(rows))

    analyzed = sum(
        1 for item in predictions.values()
        if item.get("last_updated") != "Not yet predicted"
    )

    col2.metric("Records With Predictions", analyzed)

    st.dataframe(df, use_container_width=True, hide_index=True)

    st.subheader("Latest Detection Images")

    latest_images = read_json(
        DATA_DIR / "latest_images.json", {}
    )

    for username, image_path in latest_images.items():
        if image_path and Path(image_path).is_file():
            with st.expander(f"View image: {username}"):
                st.image(str(image_path), use_container_width=True)


# ==========================================================
# SAVE DETECTION RESULTS
# ==========================================================

def process_image(image, source_name):
    model = check_model()

    if model is None:
        return

    try:
        with st.spinner("Analyzing image with YOLO..."):
            results = model.predict(
                source=image,
                imgsz=640,
                conf=0.25,
                verbose=False
            )

        result = results[0]

        annotated_bgr = result.plot()
        annotated_rgb = cv2.cvtColor(
            annotated_bgr, cv2.COLOR_BGR2RGB
        )

        st.subheader("Detection Result")
        st.image(
            annotated_rgb,
            caption="YOLO Detection Output",
            use_container_width=True
        )

        detections = []

        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                label = result.names[class_id]

                detections.append(
                    f"{label} (confidence: {confidence:.2f})"
                )

        if detections:
            st.subheader("Detected Classes")

            for detection in detections:
                st.write(f"- {detection}")
        else:
            st.info("No objects were detected.")

        username = st.session_state.username
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        safe_username = "".join(
            c if c.isalnum() or c in "-_" else "_"
            for c in username
        )

        image_path = (
            PREDICTIONS_DIR
            / f"{safe_username}_{timestamp}.jpg"
        )

        cv2.imwrite(
            str(image_path),
            cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
        )

        save_latest_image(username, image_path)

        predictions = load_predictions()

        predictions[username] = {
            "engagement_labels": detections,
            "last_updated": datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "source": source_name
        }

        save_predictions(predictions)

        st.success("Detection results saved successfully.")

    except Exception as error:
        st.error(f"Could not process the image: {error}")


# ==========================================================
# STUDENT PREDICTION PAGE
# ==========================================================

def prediction_page():
    if (
        not st.session_state.logged_in
        or st.session_state.role != "Student"
    ):
        st.error("Please log in with a student account.")
        return

    st.title("🤖 AI Engagement Prediction")

    st.write(
        "Upload an image from your classroom to detect "
        "the classes your trained model recognizes."
    )

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "webp"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(
            image,
            caption="Selected Image",
            use_container_width=True
        )

        if st.button("Analyze Image", use_container_width=True):
            process_image(image, "Image Upload")


# ==========================================================
# CAMERA DETECTION PAGE
# ==========================================================

def live_detection_page():
    if (
        not st.session_state.logged_in
        or st.session_state.role != "Student"
    ):
        st.error("Please log in with a student account.")
        return

    st.title("📷 Camera Detection")

    st.write(
        "Capture a picture using your browser camera "
        "to analyze it with the trained model."
    )

    camera_image = st.camera_input("Take a picture")

    if camera_image is not None:
        image = Image.open(camera_image).convert("RGB")

        if st.button("Analyze Camera Image", use_container_width=True):
            process_image(image, "Camera Capture")


# ==========================================================
# SIDEBAR NAVIGATION
# ==========================================================

def render_sidebar():
    with st.sidebar:
        st.title("🎓 Teacher Eye")
        st.caption("AI Student Monitoring")

        if st.session_state.logged_in:
            st.write(f"User: {st.session_state.username}")
            st.write(f"Role: {st.session_state.role}")

            if st.button("🏠 Home", use_container_width=True):
                switch_page("Home")

            if st.session_state.role == "Teacher":
                if st.button(
                    "📊 Student Activity",
                    use_container_width=True
                ):
                    switch_page("StudentActivity")

            elif st.session_state.role == "Student":
                if st.button(
                    "🤖 Prediction",
                    use_container_width=True
                ):
                    switch_page("Prediction")

                if st.button(
                    "📷 Camera Detection",
                    use_container_width=True
                ):
                    switch_page("LiveDetection")

            if st.button("🚪 Logout", use_container_width=True):
                logout()

        else:
            if st.button("🏠 Home", use_container_width=True):
                switch_page("Home")

            if st.button("🔐 Login", use_container_width=True):
                switch_page("Login")

            if st.button("📝 Register", use_container_width=True):
                switch_page("Register")


# ==========================================================
# MAIN APPLICATION
# ==========================================================

def main():
    render_sidebar()

    page = st.session_state.page

    if page == "Home":
        home_page()

    elif page == "Login":
        login_page()

    elif page == "Register":
        register_page()

    elif page == "StudentActivity":
        student_activity_page()

    elif page == "Prediction":
        prediction_page()

    elif page == "LiveDetection":
        live_detection_page()

    else:
        home_page()


if __name__ == "__main__":
    main()

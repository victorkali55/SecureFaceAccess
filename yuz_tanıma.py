import cv2
import os
import numpy as np
import hashlib
from datetime import datetime

# --- DIRECTORIES AND FILE PATHS ---
DATA_DIR = "data"
FACES_DIR = os.path.join(DATA_DIR, "faces")
TRAINER_FILE = os.path.join(DATA_DIR, "trainer.yml")
USER_INFO_FILE = os.path.join(DATA_DIR, "user_info.txt")
SECRET_FILE = "secret_file.txt"

# Haar Cascade file
CASCADE_FILE = 'haarcascade_frontalface_default.xml'
CASCADE_URL = 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml'

# --- CREATE REQUIRED DIRECTORIES ---
os.makedirs(FACES_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# --- CHECK AND DOWNLOAD CASCADE FILE IF MISSING ---
if not os.path.exists(CASCADE_FILE):
    print("⚠️ 'haarcascade_frontalface_default.xml' not found.")
    print("Downloading... (Internet connection required)")
    try:
        import urllib.request
        urllib.request.urlretrieve(CASCADE_URL, CASCADE_FILE)
        print("✅ Cascade file downloaded.")
    except Exception as e:
        print(f"❌ Download failed: {e}")
        print("Download manually: https://github.com/opencv/opencv/raw/master/data/haarcascades/haarcascade_frontalface_default.xml")
        exit(1)

# --- FACE DETECTION AND RECOGNITION ---
face_cascade = cv2.CascadeClassifier(CASCADE_FILE)
recognizer = cv2.face.LBPHFaceRecognizer_create()

# --- PASSWORD HASHING ---
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# --- REGISTRATION MODE ---
def register_user():
    print("\n🔐 NEW USER REGISTRATION")
    name = input("Your name: ").strip()
    password = input("Set a password: ")

    if not name or not password:
        print("❌ Name and password cannot be empty!")
        return False

    # Save user info
    with open(USER_INFO_FILE, "w", encoding="utf-8") as f:
        f.write(f"{name}\n{hash_password(password)}")

    print(f"\n📸 {name}, face registration starting. Look at the camera...")
    print("5 photos will be taken quickly. Stay still.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Camera could not be opened!")
        return False

    count = 0
    last_capture_time = 0  # in milliseconds

    while count < 5:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # Current time in milliseconds
        current_time = cv2.getTickCount() / cv2.getTickFrequency() * 1000

        # Capture if face detected and at least 300ms passed since last capture
        if len(faces) > 0 and (current_time - last_capture_time) > 300:
            for (x, y, w, h) in faces:
                face_gray = gray[y:y+h, x:x+w]
                cv2.imwrite(f"{FACES_DIR}/user_1_{count + 1}.jpg", face_gray)
                count += 1
                last_capture_time = current_time

                # Visual feedback
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, f"Capture: {count}/5", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                break  # Capture only one face

        # Progress display
        cv2.putText(frame, f"Face Capture: {count}/5", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Face Registration', frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Face registration completed.")

    # Train the model
    train_and_save()
    return True

# --- TRAINING FUNCTION ---
def train_and_save():
    images = []
    labels = []

    for filename in os.listdir(FACES_DIR):
        if filename.endswith(".jpg"):
            path = os.path.join(FACES_DIR, filename)
            gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if gray is not None:
                images.append(gray)
                labels.append(1)

    if len(images) == 0:
        print("❌ No faces found for training!")
        return False

    recognizer.train(images, np.array(labels))
    recognizer.save(TRAINER_FILE)
    print("✅ Model trained and saved.")
    return True

# --- LOGIN (RECOGNITION) MODE ---
def login_mode():
    if not os.path.exists(TRAINER_FILE) or not os.path.exists(USER_INFO_FILE):
        print("❌ No registered user found! Please register first.")
        return

    with open(USER_INFO_FILE, "r", encoding="utf-8") as f:
        lines = f.read().strip().splitlines()
        name = lines[0]
        hashed_password = lines[1]

    print(f"\n👋 {name}, please show your face and enter your password.")
    entered_password = input("Password: ")

    if hash_password(entered_password) != hashed_password:
        print("❌ Incorrect password!")
        return

    print("Looking at the camera... (Press 'q' to quit)")

    if os.path.exists(TRAINER_FILE):
        recognizer.read(TRAINER_FILE)

    cap = cv2.VideoCapture(0)
    login_successful = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            label, confidence = recognizer.predict(gray[y:y+h, x:x+w])

            if confidence < 80 and label == 1:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"Hello {name}!", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                print("✅ Face and password match! Opening file...")
                cap.release()
                cv2.destroyAllWindows()
                open_secret_file()
                login_successful = True
                return
            else:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(frame, "Unknown...", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow('Face Login', frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    if not login_successful:
        print("❌ Authentication failed.")

# --- OPEN SECRET FILE ---
def open_secret_file():
    # Create secret file if it doesn't exist
    if not os.path.exists(SECRET_FILE):
        with open(SECRET_FILE, "w", encoding="utf-8") as f:
            f.write(f"🔐 Secret File - Access: {datetime.now().strftime('%c')}\n")
            f.write("This file contains private information.\n")
            f.write("You have access. Congratulations!\n")

    # Open file (Windows)
    try:
        os.startfile(SECRET_FILE)  # Windows
    except AttributeError:
        # macOS
        os.system(f"open '{SECRET_FILE}'")
        # Linux (uncomment if needed):
        # import subprocess; subprocess.run(['xdg-open', SECRET_FILE])

# --- MAIN MENU ---
def main():
    print("🔐 FACE + PASSWORD AUTHENTICATION SYSTEM")

    if not os.path.exists(USER_INFO_FILE):
        print("First time setup. New user registration will begin.")
        register_user()
    else:
        print("1 - Login")
        print("2 - Re-register (old data will be deleted)")
        choice = input("Your choice: ")

        if choice == "1":
            login_mode()
        elif choice == "2":
            # Clear old data
            for f in os.listdir(FACES_DIR):
                os.remove(os.path.join(FACES_DIR, f))
            for f in [TRAINER_FILE, USER_INFO_FILE]:
                if os.path.exists(f):
                    os.remove(f)
            print("🔄 Old data deleted. Starting new registration...")
            register_user()
        else:
            print("❌ Invalid choice.")

if __name__ == "__main__":
    main()
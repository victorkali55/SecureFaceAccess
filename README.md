# FaceAuth-Lock 🔐

Secure file access using face + password authentication. Built with OpenCV and pure Python — no `face_recognition` or heavy dependencies.

## 🚀 How to Run (No Setup Needed!)

Just run the Python script. All files and folders are created automatically on first launch!

### ✅ What Gets Created:
- `data/` → stores user data and model
- `data/faces/` → saves your face images
- `data/trainer.yml` → trained face recognition model (LBPH)
- `data/user_info.txt` → username and hashed password
- `secret_file.txt` → the protected file (opens on success)
- `haarcascade_frontalface_default.xml` → downloaded automatically for face detection

## 🛠️ Technology
- **Face Detection**: OpenCV Haar Cascades
- **Face Recognition**: LBPH (`cv2.face.LBPHFaceRecognizer`)
- **Security**: SHA-256 password hashing
- **Offline & Lightweight**: No internet or extra AI models needed

## ▶️ How to Start
```bash
python face_auth.py

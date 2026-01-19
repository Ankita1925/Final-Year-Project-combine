# 🩺 Disease Classification & Criticality Prediction System (GUI-FYP)

Final Year Project – GUI Development  
**Tech Stack:** React.js, Flask, MongoDB, JWT, Tailwind CSS

---

## 📌 Project Overview

The **Disease Classification & Criticality Prediction System** is a clinical decision support web application designed to assist doctors and administrators in predicting disease severity and treatment/transplant decisions for major human organs:

- 🫁 Lung  
- 🫀 Heart  
- 🩺 Liver  
- 🧬 Kidney  

The system provides a **secure, role-based GUI** where users can input patient medical parameters and receive AI-driven predictions regarding disease type, criticality, and treatment recommendations.

---

## 🎯 Key Objectives

- Provide an intuitive GUI for medical data entry
- Secure authentication using JWT
- Role-based access control (Doctor / Admin)
- Modular organ-wise prediction system
- Scalable architecture for ML model integration
- Industry-level coding conventions and clean UI design

---

## 🏗️ System Architecture

### Frontend
- **React.js (Vite)**
- **Tailwind CSS (v4) – Centralized Theme**
- Axios for API communication
- Context API for authentication state
- Protected routes & role guards

### Backend
- **Flask (Application Factory Pattern)**
- **MongoDB** for user & prediction data
- **JWT Authentication**
- RESTful APIs per organ
- MVC-style separation (Controllers, Services, Models)

---

## 👥 User Roles

| Role   | Access |
|------|------|
| Doctor | Login, Dashboard, Organ Prediction Forms |
| Admin  | Admin Dashboard, User Management |
| Public | Landing Page, Login / Register |

---

## 🔐 Authentication & Security

- JWT-based authentication
- Tokens stored securely in `localStorage`
- Role embedded inside JWT claims
- Protected routes using custom `ProtectedRoute` and `RoleGuard`
- Automatic access restriction on logout

---

## 🧪 Prediction Workflow

1. User logs in (Doctor/Admin)
2. Doctor selects organ from dashboard
3. Enters medical parameters
4. Backend processes data (Dummy ML / Real ML later)
5. System returns:
   - Disease classification
   - Criticality level (LOW / MEDIUM / HIGH)
   - Treatment or transplant decision
6. Results displayed with visual indicators

---

## 📂 Project Structure

### Backend (`/backend`)
```

backend/
├── app/
│   ├── controllers/
│   ├── models/
│   ├── services/
│   ├── utils/
│   ├── extensions.py
│   └── **init**.py
├── ml_models/
├── app.py
├── config.py
└── requirements.txt

```

### Frontend (`/frontend`)
```

frontend/
├── src/
│   ├── api/
│   ├── components/
│   ├── context/
│   ├── pages/
│   │   └── organs/
│   ├── routes/
│   ├── utils/
│   ├── App.jsx
│   └── main.jsx
├── tailwind.config.js
└── index.css

````

---

## 🎨 UI & Design Principles

- Centralized Tailwind theme (config-level)
- Consistent UI across all portals
- Card-based layouts
- Risk badges (Green / Yellow / Red)
- Responsive & accessible design
- No inline styles (strict rule)

---

## 🚀 Setup Instructions

### Prerequisites
- Node.js (v18+ recommended)
- Python 3.10+
- MongoDB (local or Atlas)

---

### Backend Setup
```bash
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
````

Backend runs at:

```
http://127.0.0.1:5000
```

---

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at:

```
http://localhost:5173
```

---

## 🔄 API Endpoints (Sample)

| Method | Endpoint            | Description          |
| ------ | ------------------- | -------------------- |
| POST   | /api/auth/login     | User login           |
| POST   | /api/auth/register  | User registration    |
| POST   | /api/lung/predict   | Lung prediction      |
| POST   | /api/kidney/predict | Kidney prediction    |
| GET    | /api/health         | Backend health check |

---

## 📜 Coding Standards Followed

* Mandatory file headers
* Mandatory function headers
* Meaningful variable names
* No unreachable code
* MVC separation
* Clean, readable logic
* Industry-level React patterns

---

## 🧠 Future Enhancements

* Integration of real trained ML models
* Prediction history & analytics
* Graphical severity meters
* Patient portal
* Deployment on cloud (AWS / Render / Vercel)

---

## 👨‍💻 Author

**Pradhumnya Changdev Kalsait**
Computer Engineering – Final Year
GUI-FYP Project

---

## 📄 License

This project is developed as part of an academic final year project and is intended for educational purposes only.

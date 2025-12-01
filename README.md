# 🧠 Social Media Mental Health Prediction System  
### A Machine Learning + Flask + AWS EC2 Project

This project predicts an individual's mental health score based on their daily social media usage.  
It uses Machine Learning (Simple Linear Regression), a Flask backend, a modern web interface, and is deployed live on AWS EC2.

---

## 📌 Features

- Predicts mental health score based on **usage hours, gender, and platform**
- Uses **Simple Linear Regression** for score prediction
- Synthetic dataset generated programmatically (5000+ rows)
- Clean, modern **HTML/CSS/JS frontend**
- **Real-time predictions** without page reload
- Full cloud deployment on **AWS EC2 (Ubuntu)**
- Model stored and loaded using **joblib**
- Responsive UI with animations

---

## 📊 Machine Learning Approach

- **Algorithm:** Simple Linear Regression  
- **Libraries:**  
  - scikit-learn  
  - numpy  
  - pandas  
  - joblib  

- **Dataset:**  
  Synthetic dataset created with hours vs. mental health score using inverse relation  
  (More hours → lower score)

- **Target Variable:**  
  Mental health score (0–100 scale)

---

## 🧩 Project Structure

```
social_media_ml/
│
├── app.py                 # Flask backend + frontend
├── train_model.py         # Regression model training
├── dataset_generator.py   # Synthetic dataset creation
├── requirements.txt       # Dependencies
├── model/
│   ├── linear_model.joblib
│   └── scaler.joblib
└── project_data.csv       # Generated dataset
```

---

## 💻 Technologies Used

### **Frontend**
- HTML  
- CSS  
- JavaScript  

### **Backend**
- Flask (Python)

### **Machine Learning**
- scikit-learn  
- numpy  
- pandas  

### **Deployment**
- AWS EC2 (Ubuntu)
- SCP for file transfer
- Virtual environment (venv)

---

## 🚀 How to Run Locally

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/SaraKolas/social_media_ml_project
cd social_media_ml
```

### 2️⃣ Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Flask App
```bash
python app.py
```

Open in browser:
```
http://127.0.0.1:5000
```

---

## ☁️ Deployment on AWS EC2

1. Launch EC2 Ubuntu instance  
2. SSH into server  
3. Install Python + pip + venv  
4. Upload files using SCP  
5. Install dependencies  
6. Open port **5000** in Security Group  
7. Run app using:  
   ```bash
   python3 app.py
   ```
8. Access using EC2 Public IP:  
   ```
   http://<public-ip>:5000
   ```

---

## 📘 Future Improvements

- Add real mental health datasets  
- Use advanced ML models (Random Forest, XGBoost)  
- Add login system + database  
- Add recommendations based on user score  
- Docker + Nginx deployment  
- Mobile-friendly UI

---

## 👩‍💻 Author

**Sara K.**  
Student | Cloud Computing & Data Science Enthusiast  

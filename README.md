# 📰 News Reality Checker – AI-Powered Fake News Detection

## 📌 Overview
The **News Reality Checker** is a deep learning–based application that detects whether a news article is **real or fake**.
It combines **LSTM** and **GRU** neural networks for advanced text classification and achieves **95% accuracy**.
The backend model is developed entirely by me, while the UI is designed using **Claude AI** for a clean and intuitive user experience.

---

## ✨ Features
- **Real-Time News Verification** – Paste any news headline or full article to instantly check its authenticity.
- **Confidence Score** – Displays prediction probability for transparency.
- **Versatile Input** – Works with short headlines and long-form articles.
- **High Accuracy** – Achieves ~95% accuracy using LSTM & GRU architectures.
- **Clean UI** – Built with Claude AI for ease of use.

---

## ⚙ Tech Stack
**Backend:** Python, TensorFlow/Keras  
**Frontend:** Claude AI-generated UI  
**ML/DL Models:** LSTM, GRU  
**Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, TensorFlow/Keras  
**Dataset:** [Fake and Real News Dataset – Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

---

## 📂 Dataset Details
The dataset consists of two CSV files:
- `True.csv` – Real news articles
- `Fake.csv` – Fake news articles

Both are merged and labeled for training.

---

## 🛠 Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/news-reality-checker.git
cd news-reality-checker
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Download Dataset from Kaggle
```bash
# Place your Kaggle API credentials in kaggle.json
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

kaggle datasets download -d clmentbisaillon/fake-and-real-news-dataset
unzip fake-and-real-news-dataset.zip -d fake_news
```

### 4️⃣ Run the Training Script
```bash
python train_model.py
```

### 5️⃣ Launch the App
```bash
streamlit run app.py
```

---

## 📊 Model Training
- **Preprocessing:** Tokenization, padding, text cleaning.
- **Architecture:** Combination of LSTM and GRU layers for robust sequence learning.
- **Evaluation:** Achieved **95% accuracy** on the test set.

---

## 📸 Screenshots
<img width="1591" height="755" alt="Screenshot 2025-08-14 124427" src="https://github.com/user-attachments/assets/32c79e74-8cdd-4864-aa57-ba1adc7ddec1" />


---

## 🚀 Future Improvements
- Add multilingual news detection
- Integrate with browser extensions for one-click verification
- Deploy as a public web app

---

## 🏷 License
This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---

## 📬 Contact
**Author:** Manan Sood  
**GitHub:** [github.com/Manan79](https://github.com/Manan79)  
**LinkedIn:** [linkedin.com/in/mannan-sood-a38688253](https://linkedin.com/in/mannan-sood-a38688253)

<h1 align="center">🌸 FloriFy 🌸</h1>
<p align="center">
  <img src="https://img.shields.io/badge/Machine%20Learning-Flower%20Classification-brightgreen?style=flat-square">
  <img src="https://img.shields.io/badge/Flask-Web%20App-blue?style=flat-square">
  <img src="https://img.shields.io/github/license/visxnu/FloriFy?style=flat-square">
</p>

<p align="center">
  <b>A beautiful web application for identifying flower species using Machine Learning.</b><br>
  Upload an image 🌼 | Classify instantly ⚡ | Built with Flask + Python
</p>

---

## 🌻 Demo

🚀 Visit the live demo (if deployed) or run locally by following the instructions below!

<p align="center">
  <img src="https://github.com/visxnu/FloriFy/assets/your-screenshot-path/florify-demo.gif" width="600px">
</p>

---

## 🧠 About the Project

**FloriFy** is a flower image classifier web application built using a pre-trained machine learning model. The app takes an image of a flower as input and returns the most likely species. It’s intuitive, fast, and perfect for learning or fun!

### 🎯 Key Features

- 📸 Upload and classify flower images
- 🤖 ML model trained on image datasets (e.g., TensorFlow/Keras-based)
- 🌐 Simple and clean Flask web interface
- 🔍 Accurate predictions with visually pleasing UI

---

## 📂 Project Structure

```
FloriFy/
│
├── app.py                  # Main Flask app
├── static/                 # Static files (CSS, images)
├── templates/              # HTML templates
├── model/                  # Saved ML model
├── notebook/               # Jupyter notebooks for training
└── requirements.txt        # Dependencies
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/visxnu/FloriFy.git
cd FloriFy
```

### 2. Create a virtual environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
python app.py
```

Then open your browser and go to `http://localhost:5000` 🌐

---

## 📸 Sample Predictions

| Input Image | Prediction |
|-------------|------------|
| ![](static/samples/rose.jpg) | 🌹 Rose |
| ![](static/samples/sunflower.jpg) | 🌻 Sunflower |
| ![](static/samples/daisy.jpg) | 🌼 Daisy |

---

## 🤝 Contributing

Contributions are welcome! If you have suggestions for improvements or want to add features:

1. Fork the repo
2. Create a new branch (`git checkout -b feature-name`)
3. Make changes and commit
4. Push and open a Pull Request

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📬 Contact

Created by **[Vishnu](https://github.com/visxnu)** ✨  
Feel free to reach out for collaborations or feedback!

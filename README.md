# Handwriting-Based Gender Classification ✍️👩‍🧑

A machine learning and deep learning based project to classify **gender from handwriting samples**.  
By analyzing handwriting strokes, curves, and writing patterns, this project predicts whether the writer is **male or female**.  

---

## 📖 Table of Contents
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Results](#-results)
- [Known Issues / Bugs](#-known-issues--bugs)
- [Future Work](#-future-work)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🚀 Features
- Preprocessing of handwriting images (grayscale, thresholding, normalization)  
- Feature extraction using classical ML and deep learning  
- Model training and evaluation with CNN, SVM, and Random Forest  
- Performance metrics: accuracy, confusion matrix, prediction samples  
- Modular code structure for easy extension  

---

## 📂 Project Structure
Handwriting-based-gender-classification/
│
├── dataset/ # Handwriting samples (train/test)
├── notebooks/ # Jupyter notebooks for experiments
├── src/ # Source code (models, preprocessing, training)
│ ├── preprocessing.py
│ ├── train.py
│ ├── evaluate.py
│ └── model.py
├── results/ # Trained models, metrics, and plots
├── requirements.txt # Python dependencies
└── README.md # Project documentation

yaml
Copy code

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone git@github.com:Afnankhan8/Handwriting-based-gender-classification.git
cd Handwriting-based-gender-classification
2. Create a virtual environment (recommended)
bash
Copy code
python -m venv venv
# On Linux/Mac
source venv/bin/activate
# On Windows
venv\Scripts\activate
3. Install dependencies
bash
Copy code
pip install -r requirements.txt
🧑‍💻 Usage
Training the model
bash
Copy code
python src/train.py --dataset dataset/ --epochs 20 --batch-size 32
Evaluating the model
bash
Copy code
python src/evaluate.py --model results/best_model.pth --test dataset/test
📊 Results
Accuracy: XX% (replace with your actual results after training)

Comparison of CNN vs SVM and Random Forest

Visualizations available in results/ folder

🐛 Known Issues / Bugs
Dataset size is limited → may affect accuracy

Results can vary if dataset is unbalanced

Training requires GPU for faster performance (CPU will be very slow)

No deployment interface yet (CLI only)

👉 If you find more issues, please create an Issue in the repo.

🔮 Future Work
Collect larger and more diverse handwriting datasets

Try advanced deep learning models (Transformers)

Add deployment via Flask/Streamlit web app

Extend classification to age group and handedness (left/right)

🤝 Contributing
Contributions are welcome!

Fork this repository

Create a new branch (git checkout -b feature-xyz)

Commit your changes (git commit -m "Add new feature")

Push to your branch (git push origin feature-xyz)

Open a Pull Request

📜 License
This project is licensed under the MIT License. You are free to use, modify, and distribute it with proper attribution.

🙌 Acknowledgements
Inspired by research on handwriting biometrics

Built with Python, OpenCV, scikit-learn, TensorFlow/PyTorch

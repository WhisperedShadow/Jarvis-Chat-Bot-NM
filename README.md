# 🤖 ChatBot using Flask, PyTorch, and NLP

This is a simple AI chatbot built with **Flask**, **PyTorch**, and **NLTK**. It understands basic user inputs and returns appropriate responses using a trained neural network model.

## 🚀 Features

- Tokenization and bag-of-words for text preprocessing
- Trained neural network model using PyTorch
- Web interface using Flask
- Wikipedia integration for general knowledge
- JSON-based intents for dialogue handling

## 📁 Project Structure
```bash
project/
│
├── static/ # Static assets (if any)
├── templates/ # HTML files (e.g., index.html)
├── intents.json # Intent definitions
├── model.pth # Trained PyTorch model
├── model.py # Neural network architecture
├── nltk_utils.py # Tokenizer and BOW functions
├── chat.py # Logic to get chatbot responses
├── app.py # Flask application
└── requirements.txt # Python dependencies

```

## ⚙️ How to Run

1. **Clone the repository**
```bash
   git clone https://github.com/whisperedshadow/jarvis-chatbot-nm.git
   cd jarvis-chatbot-nm
```
Install dependencies

```bash

pip install -r requirements.txt
Download NLTK data
```

```bash
import nltk
nltk.download('punkt')
Run the Flask app
```

```bash

python app.py

```


**Open in browser**
Visit http://127.0.0.1:5000/ in your web browser.

**📚 Dependencies**
Flask

PyTorch

NLTK

Wikipedia

NumPy
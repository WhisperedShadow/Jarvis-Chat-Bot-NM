from flask import Flask, request, jsonify, render_template
from chat import get_response

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    text = request.json.get("message")
    response = get_response(text)
    return jsonify({"reply": response})

if __name__ == "__main__":
    app.run(debug=True)

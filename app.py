from flask import Flask, render_template, request, jsonify
import pandas as pd
from chatbot_training import bot_response

data = pd.read_csv('data_clean.csv')

app = Flask(__name__)

@app.get("/")
def index_get():
    return render_template("base.html")


@app.post("/predict")
def predict():
    text = request.get_json().get("message")
    # TODO: check if text is valid
    response = bot_response(text)
    message = {"answer" : response}
    return jsonify(message)

if __name__ == "__main__":
    app.run(host ="0.0.0.0", port=8080, debug=True)
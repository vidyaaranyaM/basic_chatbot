from flask import Flask, render_template, request, jsonify
from chat import ChatBot

app = Flask(__name__)

@app.get("/")
def index_get():
    return render_template("base.html")

@app.post("/predict")
def predict():
    bot = ChatBot()
    text = request.get_json().get("message")
    response = bot.get_response(text)
    message = {"answer": response}
    return jsonify(message)

if __name__ == "__main__":
    app.run(debug=True)

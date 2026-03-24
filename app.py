from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

model = pickle.load(open("wine_model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    features = [float(x) for x in request.form.values()]
    final = [np.array(features)]
    
    prediction = model.predict(final)

    return render_template(
        "index.html",
        prediction_text="Predicted Wine Quality: {}".format(prediction[0])
    )

if __name__ == "__main__":
    app.run(debug=True)
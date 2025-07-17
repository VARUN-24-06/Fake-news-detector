from flask import Flask, render_template, request
import pickle

# Load the model
with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Load the vectorizer
with open("vectorizer.pkl", "rb") as vec_file:
    vectorizer = pickle.load(vec_file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        news = request.form['news']
        data = [news]
        vect = vectorizer.transform(data)
        prediction = model.predict(vect)
        result = "REAL News ✅" if prediction[0] == 1 else "FAKE News ❌"
        return render_template("index.html", prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
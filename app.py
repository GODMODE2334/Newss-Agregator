from flask import Flask, request, render_template
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the pre-trained TF-IDF vectorizer and model
vectorizer = pickle.load(open("vectorizer.pkl", 'rb'))
model = pickle.load(open("finalized_model.pkl", "rb"))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/prediction', methods=["GET", "POST"])
def prediction():
    if request.method == "POST":
        news = request.form['news']
        print(news)
        # Transform the input using the loaded vectorizer
        transformed_news = vectorizer.transform([news])
        # Predict using the loaded model
        prediction = model.predict(transformed_news)
        print(prediction)
        # Return the prediction result to the template
        result = "Fake News" if prediction[0] == 1 else "Real News"
        return render_template("prediction.html", prediction_text=f"News headline is: {result}")
    else:
        return render_template("prediction.html")

if __name__ == '__main__':
    app.run(debug=True)

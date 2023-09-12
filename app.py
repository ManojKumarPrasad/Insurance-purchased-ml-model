
import pickle
import numpy as np
from flask import Flask, render_template, request

# Load the pickled model
pickled_model = pickle.load(open('lr.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # For rendering results on HTML GUI
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = pickled_model.predict(final_features)
    output = int(prediction)
    print(prediction)
    if output == 1:
        return render_template('index.html',prediction_text="Wo Kharidega")
    else:
        return render_template('index.html', prediction_text="Nahi")

if __name__ == '__main__':
    app.run(debug=True)


from flask import Flask, request, render_template
import pickle
import numpy as np


model = pickle.load(open('lr_model_pickle', 'rb'))
app = Flask(__name__)


@app.route('/')         # root URL
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():                              # Take input by POST request
    area = request.form.get('area')
    bedroom = request.form.get('bedroom')
    age = request.form.get('age')

    input_query = np.array([[int(area), int(bedroom), int(age)]])
    result = model.predict(input_query)[0]

    return render_template('home.html', predict_text=f"Predicted Price by Model is {int(result)}")


if __name__ == '__main__':
    app.run(debug=True)
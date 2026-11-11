
from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('iris_model.pkl')


@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None

    if request.method == 'POST':

        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])

        input_data = [[
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]]

        # Make prediction
        result = model.predict(input_data)

        prediction = int(result[0])

    return render_template(
        "index.html",
        prediction=prediction
    )


if __name__ == '__main__':
    app.run(debug=True)

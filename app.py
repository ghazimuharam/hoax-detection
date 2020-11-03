from logging import log
from core.torch_utils import make_prediction
from flask import Flask, request, jsonify, render_template
# from core.torch_utils import make_prediction
app = Flask(__name__, template_folder='./views')


@app.route('/')
def home():
    return render_template('index.html')
    return jsonify({"status": 200, "message": "Muhammad Ghazi Muharam @2020"})


@app.route('/api/v1/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
    except:
        return jsonify({"status": 503, "message": "an error occured"})
    data = request.get_json()
    logits, label, predicted_label = make_prediction(data['text_narration'])
    prediction = {"status": 200, "text_narration": data['text_narration'], "prediction": {
        "logits": logits, "label": label, "predicted_label": predicted_label}}
    return jsonify(prediction)


if __name__ == '__main__':
    app.run(port=5000, debug=True)

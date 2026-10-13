import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

X = np.array([
    [50, 1], [60, 2], [80, 3], [100, 3], [120, 4],
    [45, 1], [55, 2], [90, 3], [110, 4], [130, 5]
], dtype=float)

y_reg = np.array([50, 65, 90, 120, 150, 45, 60, 100, 130, 160], dtype=float)
y_clf = (y_reg >= 100).astype(float)

X_norm = X / 150.0
y_reg_norm = y_reg / 200.0
y_clf = y_clf.reshape(-1, 1)
y_reg_norm = y_reg_norm.reshape(-1, 1)

np.random.seed(42)
input_size = 2
hidden_size = 4

W1 = np.random.randn(input_size, hidden_size) * 0.1
b1 = np.zeros((1, hidden_size))
W_reg = np.random.randn(hidden_size, 1) * 0.1
b_reg = np.zeros((1, 1))
W_clf = np.random.randn(hidden_size, 1) * 0.1
b_clf = np.zeros((1, 1))


def relu(x):
    return np.maximum(0, x)


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


lr = 0.1
for epoch in range(1500):
    hidden = relu(np.dot(X_norm, W1) + b1)
    pred_reg = np.dot(hidden, W_reg) + b_reg
    pred_clf = sigmoid(np.dot(hidden, W_clf) + b_clf)

    err_reg = pred_reg - y_reg_norm
    err_clf = pred_clf - y_clf

    W_reg -= lr * np.dot(hidden.T, err_reg) / len(X)
    b_reg -= lr * np.sum(err_reg, axis=0, keepdims=True) / len(X)
    W_clf -= lr * np.dot(hidden.T, err_clf * pred_clf * (1 - pred_clf)) / len(X)
    b_clf -= lr * np.sum(err_clf * pred_clf * (1 - pred_clf), axis=0, keepdims=True) / len(X)

print("Нейросеть успешно обучена!")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        area = float(request.form.get('area'))
        rooms = float(request.form.get('rooms'))

        user_input = np.array([[area, rooms]]) / 150.0

        hidden = relu(np.dot(user_input, W1) + b1)
        r_pred = (np.dot(hidden, W_reg) + b_reg)[0, 0] * 200.0
        c_pred = sigmoid(np.dot(hidden, W_clf) + b_clf)[0, 0]

        class_label = "Премиум сегмент" if c_pred >= 0.5 else "Эконом сегмент"

        return jsonify({
            'price': round(float(r_pred), 2),
            'class_label': class_label,
            'probability': round(float(c_pred) * 100, 2)
        })
    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True, port=5000)
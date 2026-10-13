import numpy as np
from flask import Flask, render_template, request, jsonify
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)


print("Загрузка датасета California Housing...")
housing = fetch_california_housing()
X_raw = housing.data
y_reg_raw = housing.target


X = X_raw[:2000]
y_reg = y_reg_raw[:2000]


median_price = np.median(y_reg)
y_clf = (y_reg >= median_price).astype(float)


scaler_X = StandardScaler()
X_norm = scaler_X.fit_transform(X)

y_reg_norm = y_reg / 5.0
y_reg_norm = y_reg_norm.reshape(-1, 1)
y_clf = y_clf.reshape(-1, 1)


np.random.seed(42)
input_size = 8
hidden_size = 16

# Веса и смещения
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


print("Обучение многозадачной нейросети на реальных данных...")
lr = 0.05
for epoch in range(800):

    hidden = relu(np.dot(X_norm, W1) + b1)

    pred_reg = np.dot(hidden, W_reg) + b_reg
    pred_clf = sigmoid(np.dot(hidden, W_clf) + b_clf)


    err_reg = pred_reg - y_reg_norm
    err_clf = pred_clf - y_clf


    W_reg -= lr * np.dot(hidden.T, err_reg) / len(X)
    b_reg -= lr * np.sum(err_reg, axis=0, keepdims=True) / len(X)

    W_clf -= lr * np.dot(hidden.T, err_clf * pred_clf * (1 - pred_clf)) / len(X)
    b_clf -= lr * np.sum(err_clf * pred_clf * (1 - pred_clf), axis=0, keepdims=True) / len(X)

print("Нейросеть успешно обучена на California Housing!")



@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:

        features = [
            float(request.form.get('f0', 8.3)),
            float(request.form.get('f1', 41.0)),
            float(request.form.get('f2', 6.9)),
            float(request.form.get('f3', 1.0)),
            float(request.form.get('f4', 322.0)),
            float(request.form.get('f5', 2.5)),
            float(request.form.get('f6', 37.88)),
            float(request.form.get('f7', -122.23))
        ]


        user_input = scaler_X.transform(np.array([features]))


        hidden = relu(np.dot(user_input, W1) + b1)
        r_pred = (np.dot(hidden, W_reg) + b_reg)[0, 0] * 5.0 * 100000  # Переводим в доллары
        c_pred = sigmoid(np.dot(hidden, W_clf) + b_clf)[0, 0]

        class_label = "Премиум сегмент (дороже медианы)" if c_pred >= 0.5 else "Бюджетный сегмент"

        return jsonify({
            'price': round(float(r_pred), 2),
            'class_label': class_label,
            'probability': round(float(c_pred) * 100, 2)
        })
    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True, port=5000)
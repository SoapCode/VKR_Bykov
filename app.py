from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__, template_folder="templates") # инициализируем фласк

@app.route('/')
def home():
    return render_template('home.html') # домашняя страничка

@app.route('/predict/', methods=['GET','POST']) # для передачи параметров из веб интерфейса

def predict():
    if request.method == "POST":
        sepal_length = request.form.get('sepal_length') # получаем параметр
        sepal_width = request.form.get('sepal_width') # получаем параметр
        petal_length = request.form.get('petal_length') # получаем параметр
        petal_width = request.form.get('petal_width') # получаем параметр

        try:
            prediction = classify(sepal_length, sepal_width, petal_length, petal_width) # получаем предсказание от модели
            return render_template('predict.html', prediction=prediction) # отправляем результат на веб-страницу
        except:
            return 'Введите допустимые значения'


def classify(a, b, c, d):
    classes = ['Setosa', 'Versicolor' , 'Virginica']
    with open('forest_model.pkl', 'rb') as file:
        model = pickle.load(file) # загружаем натренированную модель
    to_predict = np.array([a, b, c, d]).reshape(1, -1) # преобразуем для подачи в модель
    prediction = classes[model.predict(to_predict)[0]] # делаем предсказание
    return prediction # возвращаем результат

if(__name__=='__main__'): 
    app.run(debug=True) # запуск сервера
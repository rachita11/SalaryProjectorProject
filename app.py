from flask import Flask,request,url_for, render_template
import pickle
import numpy as np

app = Flask('__name__')
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predicting():
    features = [[int(i) for i in request.form.values()]]
    features = np.array(features)
    pred_val = model.predict(features)

    return render_template('index.html',results = "Your predicted salary is Rs {0:.2f}".format(pred_val[0]))


if __name__=="__main__":
    app.run(debug=True)
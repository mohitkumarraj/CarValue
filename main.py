from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
app = Flask(__name__)
#model before imputed data
# model=pickle.load(open('ln_wl_model.pkl','rb'))
# model=pickle.load(open('rf_wl_model.pkl','rb')) #random forest model without taking log value of price
model=pickle.load(open('rfmodel.pkl','rb'))#random forest model after taking log value of price

#model after imputed data
# model=pickle.load(open('ln2_wl_model.pkl','rb'))#Linear2(imputed data) model without taking log value of price
# model=pickle.load(open('ln2model.pkl','rb'))#Linear2(imputed data) model after taking log value of price
model=pickle.load(open('rfmodel.pkl','rb'))#randomforest2(imputed data) model after taking log value of price





@app.route('/')
def home():
    return render_template('index.html')

@app.route('/result2',methods=['POST'])

def result2():
    feature=[float(x) for x in request.form.values()]
    print(feature)
    final_feature=[np.array(feature)]
    prediction=model.predict(final_feature)
    price=2.718281828459**prediction
    price=np.round(price,2)
    return render_template('index.html',prediction_text='Price of car will be {}'.format(price)+'\u20B9')


app.run(debug=True)





# import numpy as np
# from flask import Flask, request, jsonify, render_template
# import pickle
# # Press Shift+F10 to execute it or replace it with your code.
# # Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
# app = Flask(__name__)
# model = pickle.load(open('model.pkl', 'rb'))
#
# @app.route('/')
#
# def home():
#     return render_template('index2.html')
#
# def ValuePredictor(to_predict_list):
#     to_predict = np.array(to_predict_list).reshape(1, 12)
#     loaded_model = pickle.load(open("model.pkl", "rb"))
#     result = loaded_model.predict(to_predict)
#     return result[0]
#
#
# @app.route('/result2', methods=['POST'])
# def result():
#     if request.method == 'POST':
#         to_predict_list = request.form.to_dict()
#         to_predict_list = list(to_predict_list.values())
#         to_predict_list = list(map(int, to_predict_list))
#         result = ValuePredictor(to_predict_list)
#
#         if int(result) == 1:
#             prediction = 'Income more than 50K'
#         else:
#             prediction = 'Income less that 50K'
#         return render_template('index.html', prediction_text='your {}'.format(prediction))
#
#         # return render_template("result.html", prediction=prediction)
#
#
# app.run(debug=True)
#









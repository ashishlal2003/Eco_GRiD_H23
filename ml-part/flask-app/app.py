import pickle
from flask import Flask, request, render_template
import pandas as pd
from waitress import serve

app = flask(__name__)

lgbmMpdel = pickle.load(open('Eco_GRiD_H23\\ml-part\\best_lgbm_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html', prediction_text='')


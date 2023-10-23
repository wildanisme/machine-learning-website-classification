import os, unicodedata
import argparse

from flask import Flask
from flask import redirect
from flask import render_template
from flask import request
from flask import url_for
from bs4 import BeautifulSoup as bs
from matplotlib.pyplot import title
import requests 

from flask_bootstrap import Bootstrap
# library untuk visualisasi
import numpy as np

# Library ngrok tunnel

import pandas as pd
import pickle
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import word_tokenize



app = Flask(__name__, static_folder='static')
bootstrap = Bootstrap(app)
app.config['UPLOAD_FOLDER'] = 'static'

# Loading model to compare the results
model = pickle.load(open("model/model.pkl", "rb"))

# load tfidf
with open("data/tfidf.txt", "rb") as f:
    tfidf = pickle.loads(f.read())

# PREPROCESSING
#cleaning
def cleaning(text):
    #remove non-ascii
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')  
    #remove number
    text = re.sub(r"\d+", "", text)
    #to lower
    text = text.lower()
    #remove punctuations
    text = re.sub(r'[^\w]|_',' ',text)
    # menghapus spasi awal dan akhir
    text = text.strip()
    #Remove additional white spaces
    text = re.sub(r'[\s]+', ' ', text)
    return text

# Mendefinisikan library stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Mendefinisikan library lemmatizer
lemmatizer = WordNetLemmatizer() # Menggunakan NLTK

#function untuk menghapus stopword
def removeStopword(text):
    # full file path
    stop_words = set(stopwords.words('/Users/wildanisme/jupyter_lab/website-classification/web/data/stopword'))
    word_tokens = word_tokenize(text)
    filtered_sentence = [w for w in word_tokens if not w in stop_words] 
    return ' '.join(filtered_sentence)

def preprocessing(text):
    text = cleaning(text) # cleaning dataset
    text = removeStopword(text) # stopword
    text = stemmer.stem(text) # stemming kata
    text = lemmatizer.lemmatize(text) # lemmatization kata
    return text


@app.route('/', methods=['GET', 'POST'])
def index(domain=None):
    if request.method == 'POST':
        domain = request.form.get('domain')
    return render_template('home.html',
                           site='Home',
                           title='Tugas Akhir Program Studi Ilmu Komputer FMIPA Universitas Pakuan',
                           domain=domain)


# prediksi url yang dimasukkan

@app.route('/prediksi', methods=['POST'])
def predict():
    if request.method == 'POST':
        domain = request.form.get('domain')
        domain = 'http://'+domain
        try:
          url_input = requests.get(domain)
          scrape = bs(url_input.content, 'html.parser')
          page = scrape

          # mengambil konten pada web
          content_page = page.get_text(separator=" ", strip=True)

          # mengambil title pada page
          title_page = page.title.string
          print(f"Judul : {title_page}")

          # Preprocessing
          prep_text = preprocessing(str(title_page.string))
          print(f"Preprocessing : {prep_text}")

          # feature engineer
          X_vec_test = tfidf.transform([prep_text])
          print(f"TF-IDF : \n{X_vec_test}")

          # prediksi model
          pred_test = model.predict(X_vec_test)[0]
          print(f"Prediksi oleh Model : {pred_test}")
          
          prob = model.predict_proba(X_vec_test)
          print(f"Probabilitas : {prob}")
          print(f"Nilai Akurasi Prediksi : {max(prob[0])}")
          max_prob = int(max(prob[0]) * 100)
        #   dump()

        #   if max(prob[0]) <= 0.75 :
        #     pred_test =  "Tidak Diketahui"

        except:
          pred_test =  "Tidak Dapat Mengekstrak Domain"
          title_page =  "Halaman yang Anda Cari Tidak Ditemukan"

    return render_template('predict.html',
                           site='Hasil Periksa',
                           title='Tugas Akhir Program Studi Ilmu Komputer FMIPA Universitas Pakuan',
                           domain=domain,
                           page=title_page,
                           content_page = content_page,
                           predict = pred_test,
                           prep_text=prep_text,
                           tfidf=X_vec_test,
                           max_prob=max_prob)


if __name__ == '__main__':
   app.run(debug = True, host='0.0.0.0')

from flask import Flask, render_template, request, jsonify
import nltk
import pickle
from nltk.corpus import stopwords
import re
from nltk.stem.porter import PorterStemmer
app = Flask(__name__)
ps = PorterStemmer()

# Load model and vectorizer
model = pickle.load(open('Fake_News_SVC.pickle', 'rb'))
tfidfvect = pickle.load(open('tfidfvect2.pkl', 'rb'))

# Build functionalities
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')
    
def predict(text):
    review = re.sub(r'[^A-Za-z0-9]+',' ', text)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    review_vect = tfidfvect.transform([review]).toarray()
    prediction = 'True News' if model.predict(review_vect) == 0 else 'Fake News'
    return prediction

@app.route('/', methods=['POST'])
def webapp():
    text = request.form['text']
    prediction = predict(text)
    return render_template('index.html', text=text, result=prediction)

@app.route('/predict/', methods=['GET','POST'])
def api():
    text = request.args.get("text")
    prediction = predict(text)
    return jsonify(prediction=prediction)

if __name__ == "__main__":
    app.run()
||||||| 714302d
import numpy as np
from flask import Flask, render_template, request, jsonify
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from nltk.corpus import stopwords
import re
from nltk.stem.porter import PorterStemmer
from markupsafe import Markup
from nltk.stem import WordNetLemmatizer
import os
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
import pandas as pd
from sklearn.model_selection import train_test_split

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

app = Flask(__name__)
data = pd.read_csv('train.csv')
df_ = data.copy()

features_dropped = ['author','title','id']
df = df_.drop(features_dropped, axis =1)

with open('model.pickle', 'rb') as handle:
	model = pickle.load(handle)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

df['text'] = df['text'].fillna('') 
def clean_text(text):
     text = re.sub(r'[^A-Za-z0-9]+',' ',text)
     text = text.lower()
     text = [lemmatizer.lemmatize(token) for token in text.split(" ")]
     text = [lemmatizer.lemmatize(token) for token in text]   # splits and single word into smaller piece
     text = [word for word in text if not word in stop_words] 
     text = " ".join(text)    
     return text

df['text'] = df.text.apply(lambda x: clean_text(x))
X = df['text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(X_train)
xv_test = vectorization.transform(X_test)

def manual_testing(news):    
    testing_news = {"text":[news]}            
    new_def_test = pd.DataFrame(testing_news)    
    new_def_test["text"] = new_def_test["text"].apply(clean_text) 
    new_x_test = new_def_test["text"] 
    New_xv_test = vectorization.transform(new_x_test)
    prediction = model.predict([New_xv_test])
    return render_template('index.html', prediction_text='The news is "{}"'.format(prediction[0]))

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        pred = manual_testing(message)
        print(pred)
        return render_template('index.html', prediction=pred)
    else:
        return render_template('index.html', prediction="Something went wrong")


if __name__=="__main__":
    port=int(os.environ.get('PORT',5000))
    app.run(port=port,debug=True,use_reloader=False)
=======
from flask import Flask, render_template, request, jsonify
import nltk
import pickle
from nltk.corpus import stopwords
import re
from nltk.stem.porter import PorterStemmer
app = Flask(__name__)
ps = PorterStemmer()

# Load model and vectorizer
model = pickle.load(open('Fake_News_SVC.pickle', 'rb'))
tfidfvect = pickle.load(open('tfidfvect2.pkl', 'rb'))

# Build functionalities
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')
    
def predict(text):
    review = re.sub(r'[^A-Za-z0-9]+',' ', text)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    review_vect = tfidfvect.transform([review]).toarray()
    prediction = 'True News' if model.predict(review_vect) == 0 else 'Fake News'
    return prediction

@app.route('/', methods=['POST'])
def webapp():
    text = request.form['text']
    prediction = predict(text)
    return render_template('index.html', text=text, result=prediction)

@app.route('/predict/', methods=['GET','POST'])
def api():
    text = request.args.get("text")
    prediction = predict(text)
    return jsonify(prediction=prediction)

if __name__ == "__main__":
    app.run()

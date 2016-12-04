from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators
import pickle
import sqlite3
import os
import numpy as np

app = Flask(__name__)

######## Preparing the Classifier
cur_dir = os.path.dirname(__file__)
clf = pickle.load(open(os.path.join(cur_dir,
                 'pkl_objects/model.pkl'), 'rb'))
db = os.path.join(cur_dir, 'input_text.sqlite')

def classify(document):
    label = {0: 'Democrat', 1: 'Republican'}
    y = clf.predict(document)
    proba = np.max(clf.predict_proba(document))
    return label[y], proba

def sqlite_entry(path, document, y):
    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.execute("INSERT INTO input_text_db (text, prediction, data)"\
    " VALUES (?, ?, DATETIME('now'))", (document, y))
    conn.commit()
    conn.close()

####### App Nuts and Bolts
app = Flask(__name__)
class InputForm(Form):
    inputtext = TextAreaField('',
                              [validators.DataRequired(),
                              validators.length(min=15)])

@app.route('/')
def index():
    form = InputForm(request.form)
    return render_template('inputform.html', form=form)

@app.route('/results', methods=['POST'])
def results():
    form = InputForm(request.form)
    if request.method == 'POST' and form.validate():
        usertext = request.form['inputtext']
        y, proba = classify(usertext)
        return render_template('results.html',
                                content=usertext,
                                prediction=y,
                                probability=round(proba*100, 2))
    return render_template('results.html', form=form)

@app.route('/thanks', methods=['POST'])
def feedback():
    feedback = request.form['feedback_button']
    review = request.form['usertext'] #double check this line
    prediction = request.form['prediction']

    inv_label = {'Democrat': 0, 'Republican': 1}
    y = inv_label[prediction]
    if feedback == 'Incorrect':
        y = int(not(y))
    sqlite_entry(db, review, y)
    return render_template('thanks.html')

if __name__ == '__main__':
    app.run(debug=True)

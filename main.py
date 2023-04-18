from flask import Flask, render_template,send_file, redirect
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
import os
from wtforms.validators import InputRequired
import keras
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.image

 
app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'static/files'
root1=r'static\files'

model = YOLO('best.pt')

class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Upload File")

@app.route('/', methods=['GET',"POST"])
def Index():
    return render_template("index123.html")

@app.route('/home', methods=['GET',"POST"])
def home():
    form = UploadFileForm()
    if form.validate_on_submit():
        global file1
        file = form.file.data # First grab the file
        file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config['UPLOAD_FOLDER'],secure_filename(file.filename)))
        file1 = file.filename
        return redirect('/Predicted')
    return render_template('index.html', form=form)

@app.route('/Predicted')
def Predicted_Page():
    model2 = keras.models.load_model('./model_sev.h5')
    model = keras.models.load_model('./abc.h5')
    image = cv2.imread(root1+'\\'+file1)
    image2 = cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)
    image2 = image2.reshape(-1,224,224,3)
    image = cv2.resize(image, (250, 250), interpolation=cv2.INTER_CUBIC)
    image = image.reshape(-1,250,250,3)
    pred2 = model2.predict(image2)
    pred = model.predict(image)
    arg2 = np.argmax(pred2)
    if arg2==0:
        arg2='Minor'
    elif arg2==1:
        arg2='Moderate'
    else:
        arg2='Severe'

    arg = np.argmax(pred)
    if arg==0:
        arg='Damaged'
    else:
        arg='Not Damaged'

    results = model(image)
    matplotlib.image.imsave('static/files/name.png', results)
    pic1 = os.path.join(app.config['UPLOAD_FOLDER'], 'name.png')
    return render_template('Result.html', pred=arg, pic1=pic1, pred1 = arg2)



if __name__ == '__main__':
    app.run(debug=True)
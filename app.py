import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, flash, request, redirect, render_template
from keras.models import load_model
from werkzeug.utils import secure_filename

model = load_model('model.h5')
ALLOWED_EXTENSIONS = {'jpg', 'jpeg'}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"

if not os.path.exists(app.config["UPLOAD_FOLDER"]):
    os.makedirs(app.config["UPLOAD_FOLDER"])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def imgPreProcess(ORIGINAL_IMAGE):
    img = cv2.imread(ORIGINAL_IMAGE)
    resized_img = cv2.resize(img, (224,224))
    normalized_resized_img = resized_img/255
    preprocessed_image = normalized_resized_img.reshape(1,224,224,3)
    return preprocessed_image

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/classificador", methods=["GET", "POST"])
def classificador():
    if request.method == "POST":
        file = request.files["file"]
        if file.filename == '':
            flash('Não foi feito upload de nenhuma imagem')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image = imgPreProcess(image_path)
            prediction = model.predict(image)
            class_idx = np.argmax(prediction, axis=1)[0]
            prob = round(prediction.max()*100, 2)
            
            return render_template('result.html', image_path=image_path, class_idx=class_idx, prob=prob)
        else:
            return "Erro ao fazer upload da imagem."
    else:
        return "Erro ao fazer upload da imagem."

if __name__  == "__main__":
    app.run(debug=True)

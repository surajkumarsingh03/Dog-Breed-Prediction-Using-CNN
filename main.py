from flask import Flask, render_template, request
import numpy as np
import os
import cv2
import json
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.models import Sequential
from keras.layers import GlobalAveragePooling2D, Dense
from keras.preprocessing import image

# Load dog names
with open('data/dog_names.json') as json_file:
    dog_names = json.load(json_file)

# Load models
ResNet50_model_for_dog_breed = ResNet50(weights='imagenet')
Res_model_for_adjusting_shape = ResNet50(weights='imagenet', include_top=False)

# Load bottleneck features and final classification model
bottleneck_features = np.load('bottleneck_features/DogResnet50Data.npz')
train_Resnet = bottleneck_features['features']
Resnet_Model = Sequential()
Resnet_Model.add(GlobalAveragePooling2D(input_shape=train_Resnet.shape[1:]))
Resnet_Model.add(Dense(133, activation='softmax'))
Resnet_Model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
Resnet_Model.load_weights('saved_models/weights.best.Resnet.hdf5')

# Utilities
def path_to_tensor(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    return np.expand_dims(x, axis=0)

def ResNet50_predict_labels(img_path):
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model_for_dog_breed.predict(img))

def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return (151 <= prediction <= 268)

def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

def Resnet_predict_breed(img_path):
    tensor = preprocess_input(path_to_tensor(img_path))
    features = Res_model_for_adjusting_shape.predict(tensor)
    prediction = Resnet_Model.predict(features)
    return dog_names[np.argmax(prediction)]

def get_correct_prenom(word, vowels):
    return "an" if word[0].lower() in vowels else "a"

def predict_image(img_path):
    vowels = ["a", "e", "i", "o", "u"]
    if dog_detector(img_path):
        predicted_breed = Resnet_predict_breed(img_path).rsplit('.', 1)[1].replace("_", " ")
        return f"📸 It's a :  {get_correct_prenom(predicted_breed, vowels)} {predicted_breed}."
    elif face_detector(img_path):
        predicted_breed = Resnet_predict_breed(img_path).rsplit('.', 1)[1].replace("_", " ")
        return f"This photo looks like {get_correct_prenom(predicted_breed, vowels)} {predicted_breed}."
    else:
        return "No human or dog could be detected, please provide another picture."

# Flask App
IMAGE_FOLDER = 'static/'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = IMAGE_FOLDER

@app.route('/')
def upload():
    return render_template("file_upload_form.html")

@app.route('/success', methods=['POST'])
def success():
    if request.method == 'POST':
        f = request.files['file']
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
        f.save(file_path)
        result_text = predict_image(file_path)
        return render_template("success.html", name='Results after Detecting Dog Breed in Input Image',
                               img=file_path, out_1=result_text)

@app.route('/info', methods=['POST'])
def info():
    return render_template("info.html")

if __name__ == '__main__':
    app.run(host="127.0.0.1", port=8080, debug=True)

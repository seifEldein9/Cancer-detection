import numpy as np
import cv2
from tkinter import filedialog
from tkinter import Tk
from tensorflow import keras

model = keras.models.load_model('model.h5')

root = Tk()
root.withdraw()  

file_path = filedialog.askopenfilename()

if file_path:
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (50, 50))
    img = np.array(img).reshape(-1, 50, 50, 1)
    predictions = model.predict(img)
    if predictions[0][0] > predictions[0][1]:
        print("benign")
    else:
        print("malignant")
else:
    print("none")

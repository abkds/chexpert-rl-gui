#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import tkinter

from tkinter import *
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk
from keras.models import load_model
from keras.preprocessing import image

import keras
import numpy
import os
import random
import numpy as np

# random.seed(42)

top = Tk()
top.title = 'cifar10'
top.geometry('800x800')
canvas = Canvas(top, width=320,height=320, bd=0, bg='white')
canvas.grid(row=1, column=0)

WIDTH = 320
HEIGHT = 320

pathologies = [
    #'No Finding',
    #'Enlarged Cardiomediastinum',
    'Cardiomegaly',
    #'Lung Opacity',
    #'Lung Lesion',
    #'Edema',
    #'Consolidation',
    #'Pneumonia',
    'Atelectasis',
    #'Pneumothorax',
    #'Pleural Effusion',
    #'Pleural Other',
    #'Fracture'
]

model_13 = keras.applications.densenet.DenseNet121(
                                        include_top=True,
                                        weights=None,
                                        input_shape=(320, 320, 1),
                                        pooling='max',
                                        classes=2)

model_7 = keras.applications.densenet.DenseNet121(
                                        include_top=True,
                                        weights=None,
                                        input_shape=(320, 320, 1),
                                        pooling='max',
                                        classes=3)

model_13.load_weights('chexpert_weights_13.h5')
model_7.load_weights('chexpert_weights_30_March.h5')


def get_pathology_variable():
    return {key: BooleanVar() for key in pathologies}


def get_next_qimage():
    # this should be choosen on the basis of teacher rl algorithm
    img_dir = os.path.join(os.getcwd(), 'images')
    filepaths = os.listdir(img_dir)
    if '.DS_Store' in filepaths:
        filepaths.remove('.DS_Store')
    img_path = random.choice(filepaths)
    return os.path.join(img_dir, img_path)
    # img = image.load_img(os.path.join(img_dir, img_path), target_size=(WIDTH, HEIGHT))
    # return img

path_var = None

X = None
y = None

def showImg():
    img_path = get_next_qimage()

    load = Image.open(img_path)
    w, h = load.size
    # print("Widht, Height: %s %s" % (w, h))
    # print("Image path: %s" % (img_path))
    load = load.resize((WIDTH, HEIGHT))
    imgfile = ImageTk.PhotoImage(load )

    global X
    global y
    X = np.empty((1, 320, 320, 1))
    y = np.empty(1, dtype=int)

    cur_dir = os.getcwd()

    # Generate data
    #for i, path in enumerate(filepaths):
    #    img_path = os.path.join(cur_dir, path)
    img = image.load_img(img_path, target_size=(320, 320), grayscale=True)

    X[0, ] = image.img_to_array(img) / 255
    
    canvas.image = imgfile  # <--- keep reference of your image
    canvas.create_image(2, 2, anchor='nw', image=imgfile)

    global path_var
    path_var = get_pathology_variable()

    for idx, pathology in enumerate(pathologies):
        Checkbutton(top, text=pathology, variable=path_var[pathology]).grid(row=idx+2, column=1, sticky=W)


e = StringVar()

submit_button = Button(top, text ='Show Image', command = showImg)
submit_button.grid(row=0, column=0)

label_name = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']


def Submit():
    # img=Image.open(e.get())
    # img=img.resize((32, 32))
    # imgArray = numpy.array(img)
    # imgArray = imgArray.reshape(1, 32 * 32 * 3)
    # imgArray = imgArray.astype('float32')
    # imgArray /= 255.0
    # model=load_model('mlp_cifar10.h5')

    # clsimg = model.predict_classes(imgArray)
    print(path_var)

    # this we will get from the trained model
    correct_answers = {
        #'No Finding': False,
        #'Enlarged Cardiomediastinum': False,
        'Cardiomegaly': False,
        #'Lung Opacity': False,
        #'Lung Lesion': False,
        #'Edema': True,
        #'Consolidation': False,
        #'Pneumonia': True,
        'Atelectasis': False,
        #'Pneumothorax': False,
        #'Pleural Effusion': True,
        #'Pleural Other': False,
        #'Fracture': False
    }

    y_7 = model_7.predict(X)[0]
    y_13 = model_13.predict(X)[0]

    correct_answers['Atelectasis'] = y_13[1] > 0.35
    correct_answers['Cardiomegaly'] = np.argmax(y_7) == 1

    print(y_7)
    print(y_13)


    answers = []
    answer_values = []
    for pathology in pathologies:
        if path_var[pathology].get() == correct_answers[pathology]:
            answers.append('%s (correct)\n' % pathology)
            answer_values.append(False)
        else:
            answers.append('%s (incorrect)\n' % pathology)
            answer_values.append(True)

    global l2
    if any(answer_values):
        l2.config(text='You predicted wrong pathologies, please practice on more')
    else:
        l2.config(text='You are doing great, you can continue trying on more X-Rays')

    for idx, answer in enumerate(answers):
        l = Label(top, text=answer, borderwidth=1, height=1, fg="blue", wraplength=250, justify='left', anchor='n')
        l.grid(row=idx+3, column=0)



    # textvar = "The object is : %s" %(label_name[int(clsimg)])
    # t1.delete(0.0, tkinter.END)
    # t1.insert('insert', textvar+'\n')
    # t1.update()


submit_button = Button(top, text ='Submit', command = Submit)
submit_button.grid(row=0, column=1)

l1 = Label(top, text='Click on <Show Image> for a new X-Ray')
l1.grid(row=2)

l2 = Label(top,
           text='Try to predict the pathologies in the x-ray. (When you are ready check the pathologies and click submit)',
           fg="red",
           font=("Courier", 18),
           anchor='s',
           wraplength=475,
           justify='left')

l2.grid(row=1, column=1)

top.mainloop()
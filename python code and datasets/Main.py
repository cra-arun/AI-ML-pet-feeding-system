from tkinter import *
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pickle
import os
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential, load_model, Model
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import pandas as pd
import webbrowser

main = tkinter.Tk()
main.title("AI & ML Based Pet Feeding System using Image Processing")
main.geometry("1300x1200")

global filename
global X_train, y_train, X_test, y_test, labels, X, Y, cnn_model

def findLabels(path):
    global labels
    labels = []
    for root, dirs, directory in os.walk(path):
        for j in range(len(directory)):
            name = os.path.basename(root)
            if name not in labels:
                labels.append(name.strip())

def getLabel(name):
    index = -1
    for i in range(len(labels)):
        if labels[i] == name:
            index = i
            break
    return index                

def uploadDataset():
    text.delete('1.0', END)
    global filename, dataset, labels, X, Y
    filename = filedialog.askdirectory(initialdir=".")
    pathlabel.config(text=filename)
    findLabels(filename)
    if os.path.exists("model/X.npy"):
        X = np.load('model/X.npy')
        Y = np.load('model/Y.npy')
    else:
        X = []
        Y = []
        for root, dirs, directory in os.walk(filename):
            for j in range(len(directory)):        
                name = os.path.basename(root)
                if 'Thumbs.db' not in directory[j]:
                    img = cv2.imread(root+"/"+directory[j])
                    img = cv2.resize(img, (32, 32))
                    X.append(img)
                    label = getLabel(name)
                    Y.append(label) 
        X = np.asarray(X)
        Y = np.asarray(Y)
        np.save('model/X',X)
        np.save('model/Y',Y)                    
    text.insert(END,"Dataset Loading Completed\n")
    text.insert(END,"Total images found in dataset = "+str(X.shape[0])+"\n\n")
    unique, count = np.unique(Y, return_counts=True)
    for i in range(len(labels)):
        text.insert(END,"Pet = "+labels[i]+" Total Images = "+str(count[i])+"\n")

def imagePreprocessing():
    global X, Y
    text.delete('1.0', END)
    X = X.astype('float32')
    X = X/255
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    Y = to_categorical(Y)
    text.insert(END,"Dataset Shuffling & Normalization Completed")

def splitDataset():
    global X, Y
    global X_train, y_train, X_test, y_test
    text.delete('1.0', END)
    #split dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)
    text.insert(END,"Dataset Train & Test Split Details\n")
    text.insert(END,"80% dataset for training : "+str(X_train.shape[0])+"\n")
    text.insert(END,"20% dataset for testing  : "+str(X_test.shape[0])+"\n")
    data = np.load("model/data.npy", allow_pickle=True)
    X_train, X_test, y_train, y_test = data

   
def runCNN():
    global X_train, y_train, X_test, y_test
    global cnn_model
    text.delete('1.0', END)
    cnn_model = Sequential()
    cnn_model.add(Convolution2D(32, (3 , 3), input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]), activation = 'relu'))
    cnn_model.add(MaxPooling2D(pool_size = (2, 2)))
    cnn_model.add(Convolution2D(32, (3, 3), activation = 'relu'))
    cnn_model.add(MaxPooling2D(pool_size = (2, 2)))
    cnn_model.add(Flatten())
    cnn_model.add(Dense(units = 256, activation = 'relu'))
    cnn_model.add(Dense(units = y_train.shape[1], activation = 'softmax'))
    cnn_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    if os.path.exists("model/cnn_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/cnn_weights.hdf5', verbose = 1, save_best_only = True)
        hist = cnn_model.fit(X_train, y_train, batch_size = 16, epochs = 30, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
        f = open('model/cnn_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()    
    else:
        cnn_model.load_weights("model/cnn_weights.hdf5")
    predict = cnn_model.predict(X_test)
    predict = np.argmax(predict, axis=1)
    y_test1 = np.argmax(y_test, axis=1)
    p = precision_score(y_test1, predict,average='macro') * 100
    r = recall_score(y_test1, predict,average='macro') * 100
    f = f1_score(y_test1, predict,average='macro') * 100
    a = accuracy_score(y_test1,predict)*100
    algorithm = "CNN Overall"
    text.insert(END,algorithm+" Accuracy  : "+str(a)+"\n")
    text.insert(END,algorithm+" Precision : "+str(p)+"\n")
    text.insert(END,algorithm+" Recall    : "+str(r)+"\n")
    text.insert(END,algorithm+" FSCORE    : "+str(f)+"\n\n")
    report = classification_report(y_test1, predict, target_names=labels, output_dict=True)
    df = pd.DataFrame(report).transpose()
    df = df.values
    output = "<html><body><center><table align=center border=1><tr><th>Precision</th><th>Recall</th><th>FSCORE</th>"
    output += '<th>Support</tr>'
    for i in range(len(labels)):
        output += "<tr><td>"+str(labels[i])+"</td>"
        output += "<td>"+str(df[i,0])+"</td>"
        output += "<td>"+str(df[i,1])+"</td>"
        output += "<td>"+str(df[i,2])+"</td>"
        output += "<td>"+str(df[i, 3])+"</td></tr>"
    output += "</table><br/><br/><br/><br/>"    
    with open("output.html", "wb") as file:
        file.write(output.encode())
    file.close()
    webbrowser.open("output.html", new=2)

def graph():
    f = open('model/cnn_history.pckl', 'rb')
    train_values = pickle.load(f)
    f.close()
    loss = train_values['loss']
    val_loss = train_values['val_loss']
    acc = train_values['accuracy']
    val_acc = train_values['val_accuracy']
    plt.figure(figsize=(6,4))
    plt.grid(True)
    plt.xlabel('EPOCH')
    plt.ylabel('Accuracy/Loss')
    plt.plot(loss, 'ro-', color = 'green')
    plt.plot(val_loss, 'ro-', color = 'blue')
    plt.plot(acc, 'ro-', color = 'red')
    plt.plot(val_acc, 'ro-', color = 'yellow')
    plt.legend(['Training Loss', 'Validation Loss', 'Training Accuracy', 'Validation Accuracy'], loc='upper left')
    plt.title('CNN Training Accuracy & Loss Graph')
    plt.show()

def predict():
    global cnn_model, labels
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="testImages")
    img = cv2.imread(filename)
    img = cv2.resize(img, (32,32))
    im2arr = np.array(img)
    im2arr = im2arr.reshape(1,32,32,3)
    img = np.asarray(im2arr)
    img = img.astype('float32')
    img = img/255
    preds = cnn_model.predict(img)
    predict = np.argmax(preds)
    data = ""
    with open("food/"+labels[predict]+".txt", "r", encoding='utf-8') as file:
        for line in file:
            values = line.strip()
            if len(values) == 0:
                data += "\n"
            else:
                data += values+"\n"
        file.close()
    text.insert(END,data)
    text.update_idletasks()
    img = cv2.imread(filename)
    img = cv2.resize(img, (700,400))
    cv2.putText(img, 'A pet is detected | Pet Classified As: ' + labels[predict], (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(img, 'Food is dispensed', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.imshow('Pet Classified As : '+labels[predict], img)
    cv2.waitKey(0)
    
          

font = ('times', 16, 'bold')
title = Label(main, text='AI & ML Based Pet Feeding System using Image Processing',anchor=W, justify=CENTER)
title.config(bg='yellow4', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)


font1 = ('times', 13, 'bold')
upload = Button(main, text="Upload Pet Dataset", command=uploadDataset)
upload.place(x=1000,y=100)
upload.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='yellow4', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=1000,y=150)

preprocessButton = Button(main, text="Preprocess Dataset", command=imagePreprocessing)
preprocessButton.place(x=1000,y=200)
preprocessButton.config(font=font1)

splitButton = Button(main, text="Train & Test Split", command=splitDataset)
splitButton.place(x=1000,y=250)
splitButton.config(font=font1)

cnnButton = Button(main, text="Train CNN Algorithm", command=runCNN)
cnnButton.place(x=1000,y=300)
cnnButton.config(font=font1)

graphButton = Button(main, text="Training Graph", command=graph)
graphButton.place(x=1000,y=350)
graphButton.config(font=font1)

predictButton = Button(main, text="Pet Classification & Feeding System", command=predict)
predictButton.place(x=1000,y=400)
predictButton.config(font=font1)

font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=120)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=100)
text.config(font=font1)


main.config(bg='light salmon')
main.mainloop()

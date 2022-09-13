import PySimpleGUI as sg
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
size = 500, 300
from PIL import Image
import glob
import shutil
from distutils.dir_util import copy_tree

y=''
pred=0
profile=0


def get_prediction(image_to_test_path):
    mtcnn = MTCNN(image_size=240, margin=0, min_face_size=20) # initializing mtcnn for face detection
    final_classes = 10
    mid_classes = 100
    model = models.resnet18()
    num_ftrs = model.fc.in_features
    x = torch.nn.Sequential(torch.nn.Linear(in_features=num_ftrs, out_features=mid_classes, bias=False),
                            torch.nn.Linear(in_features=mid_classes, out_features=final_classes))
    model.fc = x
    model.load_state_dict(torch.load("Model_actori_romani.pth"))

    resnet = model.eval() # initializing resnet for face img to embeding conversion

    dataset=datasets.ImageFolder('testing_model')# photos folder path
    idx_to_class = {i:c for c,i in dataset.class_to_idx.items()} # accessing names of peoples from folder names


    def collate_fn(x):
        return x[0]

    loader = DataLoader(dataset, collate_fn=collate_fn)

    face_list = [] # list of cropped faces from photos folder
    name_list = [] # list of names corrospoing to cropped photos
    embedding_list = [] # list of embeding matrix after conversion from cropped faces to embedding matrix using resnet

    for img, idx in loader:
        face, prob = mtcnn(img, return_prob=True)


        if face is not None and prob>0.90: # if face detected and porbability > 90%
            emb = resnet(face.unsqueeze(0)) # passing cropped face into resnet model to get embedding matrix
            embedding_list.append(emb.detach()) # resulten embedding matrix is stored in a list
            name_list.append(idx_to_class[idx]) # names are stored in a list

    data = [embedding_list, name_list]
    torch.save(data, 'data.pt') # saving data.pt file


    def face_match(img_path, data_path):  # img_path= location of photo, data_path= location of data.pt
        # getting embedding matrix of the given img
        img = Image.open(img_path)
        face, prob = mtcnn(img, return_prob=True)  # returns cropped face and probability
        emb = resnet(face.unsqueeze(0)).detach()  # detech is to make required gradient false

        saved_data = torch.load('data.pt')  # loading data.pt file
        embedding_list = saved_data[0]  # getting embedding data
        name_list = saved_data[1]  # getting list of names
        dist_list = []  # list of matched distances, minimum distance is used to identify the person

        for idx, emb_db in enumerate(embedding_list):
            dist = torch.dist(emb, emb_db).item()
            dist_list.append(dist)

        idx_min = dist_list.index(min(dist_list))
        return (name_list[idx_min], min(dist_list))


    result = face_match(image_to_test_path, 'data.pt')
    return result[0]

#print('Face matched with: ', get_prediction()[0], 'With distance: ', get_prediction()[1])

def get_path(path):
    x1 = len(path)
    new_list = []
    for jj in path:
        new_list.append(jj)

    kounter = 0
    for i1 in range(x1 - 1, -1, -1):
        if path[i1] == "/":
            break
        kounter += 1

    newlist1 = new_list[:-kounter-1]
    file_name1 = new_list[-kounter:]
    new_path = ''
    file_name = ''
    for jjj in newlist1:
        new_path += jjj
    for jjjj in file_name1:
        file_name += jjjj

    return new_path, file_name


    # First the window layout in 2 columns

file_list_column = [
    [   sg.Text("Add a profile:"),
        sg.InputText(size=(30, 2)),
        sg.Submit("ADD")
        ],
    [
      sg.Text("Populate the profile:"),
      sg.In(size=(24, 2), enable_events=True, key="-FILE-"),
      sg.FileBrowse(key="-Populate-"),
      sg.Button("Submit picture"),


    ],

      [ sg.Text("ImageToTest"),
        sg.In(size=(30, 2), enable_events=True, key="-FOLDER-"),
        sg.FileBrowse(key="-TestPicture-"),
        sg.Button("Predict")


    ],

     [sg.Text("Prediction:"),sg.In(size=(30, 2), enable_events=True ,key='-Prediction-')],
      [sg.Cancel()],
]

# For now will only show the name of the file that was chosen
image_viewer_column = [
    [sg.Text("Your image will appear here")],
    [sg.Text(size=(40, 1), key="-TOUT-")],
    [sg.Image(key="-IMAGE-")],



]

# ----- Full layout -----
layout = [
    [
        sg.Column(file_list_column),
        sg.VSeperator(),
        sg.Column(image_viewer_column),
    ]
]

window = sg.Window("Image Viewer", layout)
event_counter=0
predictions=[]
# Run the Event Loop
jv = 0
while True:
    event, values = window.read()
    if profile==0:
        try:
            path='D:/pycharm/pytorch/pytorch_guides/Face_recog_attempt3/testing_model/'+values[0]
            os.mkdir(path)
            profile += 1
        except:
            print("No new profile added")


    if event== "Submit picture":
        try:
            file=values["-Populate-"]
            print(file)

            alabala, nume_poza=get_path(file)

            for jpgfile in glob.iglob(os.path.join(alabala, "*.jpg")):
                if jpgfile==alabala+str('\\')+nume_poza:
                    shutil.copy(jpgfile, path)
            for pngfile in glob.iglob(os.path.join(alabala, "*.png")):
                if pngfile == alabala+str('\\')+nume_poza:
                    shutil.copy(pngfile, path)
            for pngfile in glob.iglob(os.path.join(alabala, "*.jpeg")):
                if pngfile == alabala+str('\\')+nume_poza:
                    shutil.copy(pngfile, path)
        except:
                print("no path was added")

    if event == "Exit" or event == sg.WIN_CLOSED or event == "Cancel":
        try:

             #for k in range(len(file_list)-len(file_list1)):
             for k in range(1):
                   os.remove(directory_path+"//"+file_list[k],dir_fd=None)
             dir = 'D:/pycharm/pytorch/pytorch_guides/Face_recog_attempt3/testing_model/'
             for f in os.listdir(dir):
                 if f == name_of_picture:

                    os.remove(os.path.join(dir, f))

        except:
           print('nothing to remove')
        break
    # Folder name was filled in, make a list of files in the folder

    elif event == "Predict":
        file_to_test=values["-TestPicture-"]

        directory_path, name_of_picture=get_path(file_to_test)
        #toDirectory = "D:/pycharm/pytorch/pytorch_guides/data/faces2/test/aaa/"
        for jpgfile in glob.iglob(os.path.join(directory_path, "*.jpg")):

            if jpgfile == directory_path+'\\'+ name_of_picture:
                shutil.copy(jpgfile, path)
        for pngfile in glob.iglob(os.path.join(directory_path, "*.png")):
            if pngfile == directory_path+'\\'+ name_of_picture:
                shutil.copy(pngfile, path)
        for pngfile in glob.iglob(os.path.join(directory_path, "*.jpeg")):
            if pngfile == directory_path+'\\'+ name_of_picture:
                shutil.copy(pngfile, path)
        try:
            # Get list of files in folder
            file_list1 = [name_of_picture]

            #file_list2=os.listdir(toDirectory)
        except:
            file_list1 = []
            file_list2=[]

        new_file_list = []
        for element in file_list1:
            if ".jpg" in element:
                new_file_list.append(element)
            if ".png" in element:
                new_file_list.append(element)
            if ".jpeg" in element:
                new_file_list.append(element)


        if event_counter<=1:
            k=0
            for j in new_file_list:
                k+=1
                new_path = directory_path + "//" + j

                im1 = Image.open(new_path)
                im1.thumbnail(size,Image.ANTIALIAS)
                im1.save(directory_path + "//" +str(k) +'.png')

        fnames = [
            f
            for f in new_file_list
            if os.path.isfile(os.path.join(directory_path, f))
            and f.lower().endswith((".png"))
        ]


        #-------------------------------------


        try:
            # Get list of files in folder
            file_list = os.listdir(directory_path)

        except:
            file_list = []

        fnames = [
            f
            for f in file_list
            if os.path.isfile(os.path.join(directory_path, f))
            and f.lower().endswith((".png"))
        ]
        fnames2=[]
        for kj in fnames:
            try:
                fnames2.append(str(int(kj[:-4]))+'.png')
            except:
                print()

        try:
            predictions = get_prediction(file_to_test)
            print(predictions)
            window['-Prediction-'].update(predictions)
            window["-IMAGE-"].update(directory_path+"\\"+fnames2[0])
        except:
            print()

window.close()







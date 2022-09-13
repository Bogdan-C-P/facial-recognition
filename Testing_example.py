from facenet_pytorch import MTCNN

from torch.utils.data import DataLoader
from PIL import Image
import torch
import torch.nn as nn

from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt

import matplotlib.image as mpimg
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
activation = {}


def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()

    return hook
res={}
def face_match(img_path, data_path):  # img_path= location of photo, data_path= location of data.pt
    # getting embedding matrix of the given img
    img = Image.open(img_path)
    face, prob = mtcnn(img, return_prob=True)  # returns cropped face and probability
    model.fc[0].register_forward_hook(get_activation('fc_for_picture_1'))
    emb = resnet(face.unsqueeze(0)).detach()  # detech is to make required gradient false
    m = nn.Softmax(dim=1)
    #probabily, indexes = torch.max(m(emb), 1)
    probabily= m(emb)

    print(emb)
    print(probabily)
    saved_data = torch.load('data.pt')  # loading data.pt file
    embedding_list = saved_data[0]  # getting embedding data
    name_list = saved_data[1]  # getting list of names
    dist_list = []  # list of matched distances, minimum distance is used to identify the person

    for idx, emb_db in enumerate(embedding_list):
        dist = torch.dist(emb, emb_db).item()
        dist_list.append(dist)
    print(name_list)
    print(dist_list)

    idx_min = dist_list.index(min(dist_list))
    return (name_list[idx_min], min(dist_list))

image_to_test='6.jpg'
#result = face_match('morgan-freeman.jpg', 'data.pt')
result = face_match(image_to_test, 'data2.pt')
image_test = mpimg.imread(image_to_test)
plt.title("Predictie: "+ str(result[0]) +", cu distanta " +str(round(result[1],2)))
imgplot = plt.imshow(image_test)

plt.show()
#print(activation)
print('Face matched with: ', result[0], 'With distance: ', result[1])



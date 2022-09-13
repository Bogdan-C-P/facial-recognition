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

if __name__ == '__main__':

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    data_transforms2={
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([1/0.485, 1/0.456, 1/0.406], [1/0.229, 1/0.224, 1/0.225])
        ])}


    data_dir = 'actori_romani'

    image_datasets_train = datasets.ImageFolder(os.path.join(data_dir, 'train'),
                                                data_transforms['train'])

    image_datasets_test = datasets.ImageFolder(os.path.join(data_dir, 'test'),
                                                data_transforms['test'])

    batch_size = 1
    test_dataloader = torch.utils.data.DataLoader(image_datasets_test, batch_size=batch_size,
                                                   shuffle=False, num_workers=1)

    train_dataloader = torch.utils.data.DataLoader(image_datasets_train, batch_size=batch_size,
                                                   shuffle=True, num_workers=1)

    class_names = image_datasets_train.classes
    print(len(image_datasets_test))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    final_classes = len(class_names)
    mid_classes = 100
    model = models.resnet18()
    num_ftrs = model.fc.in_features
    x = torch.nn.Sequential(torch.nn.Linear(in_features=num_ftrs, out_features=mid_classes, bias=False),
                            torch.nn.Linear(in_features=mid_classes, out_features=final_classes))
    model.fc = x
    model.load_state_dict(torch.load("Model_actori_romani.pth"))
    model.to(device)

    activation = {}


    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()

        return hook

    def predictions(model):

        prediction = []
        model.eval()
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(test_dataloader):

                inputs = inputs.to(device)
                labels = labels.to(device)

                model.fc[0].register_forward_hook(get_activation('fc_for_picture'))
                outputs = model(inputs)
                print(outputs)
                _, preds = torch.max(outputs, 1)
                m = nn.Softmax(dim=1)
                probabily, indexes = torch.max(m(outputs), 1)
                for j in range(len(probabily)):
                    if probabily[j] < 0.7:
                        preds[j] = 999999
                for j in range(inputs.size()[0]):
                    if (preds[j] == 999999):
                        prediction.append('unknown')
                    else:
                        prediction.append(class_names[preds[j]])
        return prediction


    predictions(model)

    print(activation['fc_for_picture'])
    print(predictions(model))



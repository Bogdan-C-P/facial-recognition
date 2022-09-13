from operator import mod
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
    plt.ion()
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    data_dir = '../Face_recog_att4/actori_romani_augumentare'
    image_datasets_train = datasets.ImageFolder(os.path.join(data_dir, 'train'),
                                                data_transforms['train'])

    image_datasets_val = datasets.ImageFolder(os.path.join(data_dir, 'val'),
                                              data_transforms['val'])

    batch_size=8
    train_dataloader = torch.utils.data.DataLoader(image_datasets_train, batch_size=batch_size,
                                                   shuffle=True, num_workers=1)

    test_dataloader = torch.utils.data.DataLoader(image_datasets_val, batch_size=batch_size,
                                                  shuffle=True, num_workers=1)

    dataset_sizes_train = len(image_datasets_train)
    dataset_sizes_val = len(image_datasets_val)
    print(dataset_sizes_val)
    print(dataset_sizes_train)
    loss_vector_tran = []
    loss_vector_validation = []
    iters = []
    class_names = image_datasets_train.classes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    def imshow(inp, title=None):
        """Imshow for Tensor."""
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        plt.imshow(inp, cmap='gray')

        if title is not None:
            plt.title(title)
        plt.pause(0.001)  # pause a bit so that plots are updated
        plt.show()

    #inputs, classes = next(iter(train_dataloader))

    # Make a grid from batch
    #out = torchvision.utils.make_grid(inputs)

    #imshow(out, title=[class_names[x] for x in classes])

    def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        for epoch in range(num_epochs):
            iters.append(epoch)
            # Each epoch has a training and validation phase
            print("Epoch: ", epoch + 1, "/", num_epochs)
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0
                if phase == 'train':
                    for inputs, labels in train_dataloader:
            
                        inputs = inputs.to(device)
                        labels = labels.to(device)
                        
                        # zero the parameter gradients
                        optimizer.zero_grad()
                        # forward
                        # track history if only in train
                        with torch.set_grad_enabled(phase == 'train'):
                            outputs = model(inputs)
                            print('outputs',outputs)
                            print(outputs.shape)
                            print('labels',labels)
                            print(labels.shape)
                           
                            _, preds = torch.max(outputs, 1)
                            print("torchamx ",torch.max(outputs, 1))
                            print("outputs",outputs)
                            break
                    
                            loss = criterion(outputs, labels)
                            loss.backward()
                            optimizer.step()

                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == labels.data)
                    scheduler.step()
                    epoch_loss = running_loss / len(image_datasets_train)
                    loss_vector_tran.append(epoch_loss)
                    epoch_acc = running_corrects.double() / len(image_datasets_train)
                    print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                        phase, epoch_loss, epoch_acc))
                running_loss = 0.0
                running_corrects = 0
                if phase == "val":
                    for inputs, labels in test_dataloader:
                        inputs = inputs.to(device)
                        labels = labels.to(device)
                        optimizer.zero_grad()
                        with torch.no_grad():
                            outputs = model(inputs)
                            
                            _, preds = torch.max(outputs, 1)
                            
                            
                            loss = criterion(outputs, labels)
                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == labels.data)
                    epoch_loss = running_loss / len(image_datasets_val)
                    loss_vector_validation.append(epoch_loss)
                    epoch_acc = running_corrects.double() / len(image_datasets_val)
                    print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                        phase, epoch_loss, epoch_acc))
                    if epoch_acc > best_acc:
                        best_acc = epoch_acc
                        best_model_wts = copy.deepcopy(model.state_dict())
            print()
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        model.load_state_dict(best_model_wts)
        return model

    def visualize_model(model, num_images=6):
        was_training = model.training
        model.eval()
        images_so_far = 0
        fig = plt.figure()

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(test_dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                m=nn.Softmax(dim=1)
                probabily, indexes=torch.max(m(outputs),1)
                for j in range(len(probabily)):
                    if probabily[j] < 0.4:
                        preds[j] = 999999


                for j in range(inputs.size()[0]):

                    images_so_far += 1
                    ax = plt.subplot(num_images // 2, 2, images_so_far)
                    ax.axis('off')
                    if(preds[j]==999999):
                        ax.set_title('unknown')
                    else:
                        ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                    imshow(inputs.cpu().data[j])

                    if images_so_far == num_images:
                        model.train(mode=was_training)
                        return
            model.train(mode=was_training)




    model_conv = torchvision.models.resnet18(pretrained=True)
    for param in model_conv.parameters():
        param.requires_grad = False

    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = model_conv.fc.in_features

    mid_classes = 100
    final_classes = len(class_names)
    x = torch.nn.Sequential(torch.nn.Linear(in_features=num_ftrs, out_features=mid_classes, bias=False),
                            torch.nn.Linear(in_features=mid_classes, out_features=final_classes))
    model_conv.fc = x

    '''
    num_classes = len(class_names)
    model_conv.fc = nn.Linear(num_ftrs, num_classes)
    '''

    

    model_conv = model_conv.to(device)

    criterion = nn.CrossEntropyLoss()




    # Observe that only parameters of final layer are being optimized as
    # opposed to before.
    #optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.02, momentum=0.9) # acuratete 83%
    #exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
    optimizer_conv = torch.optim.Adam(model_conv.fc.parameters(), lr=0.01) #acuratete 86%
    # Decay LR by a factor of 0.1 every 5 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=5, gamma=0.3)

    model_conv = train_model(model_conv, criterion, optimizer_conv,
                             exp_lr_scheduler, num_epochs=1)


    torch.save(model_conv.state_dict(), "model_actori_romani_augumentare.pth") #acuratete 86%
    #torch.save(model_conv.state_dict(), "Model_actori_romani_1_layer.pth") #acuratete 74%%
    visualize_model(model_conv)
    print(loss_vector_validation)
    print(loss_vector_tran)
    print(iters)
    plt.ioff()
    plt.show()
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import torchvision.models as models
from collections import OrderedDict
import PIL
from PIL import Image
import json
from matplotlib.ticker import FormatStrFormatter


networks = {"vgg16": 25088,
            "densenet121": 1024,
            "alexnet": 9216,
            "resnet101": 4096}

def load_data(where  = "./flowers"):
    data_dir = where
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    data_transforms_train = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    data_transforms_validate = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
    data_transforms_test= transforms.Compose([transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406],
                                                                 [0.229, 0.224, 0.225])])

# TODO: Load the datasets with ImageFolder
    image_datasets = datasets.ImageFolder(train_dir,transform=data_transforms_train)
    image_datasets_validation =  datasets.ImageFolder(valid_dir,transform=data_transforms_validate)
    image_datasets_test =  datasets.ImageFolder(test_dir,transform=data_transforms_test)

# TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloaders = torch.utils.data.DataLoader(image_datasets,batch_size=64, shuffle=True)
    dataloaders_validation = torch.utils.data.DataLoader(image_datasets_validation,batch_size=32, shuffle=True)
    dataloaders_test = torch.utils.data.DataLoader(image_datasets_test,batch_size=20, shuffle=True)

    return dataloaders, dataloaders_validation, dataloaders_test
    
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

# TODO: Build and train your network

def nn_setup(network='vgg16', dropout=0.5, hidden_layer1=120, lr=0.001, power='gpu'):
    
    if network == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif network == 'densenet121':
        model = models.densenet121(pretrained=True)
    elif network == 'alexnet':
        model = models.alexnet(pretrained=True)
    elif network == 'resnet101':
        model = models.resnet101(pretrained=True)
    else:
        print("Im sorry but {} is not a valid model. Did you mean vgg16, densenet121 or alexnet?".format(network))

    for param in model.parameters():
        param.requires_grad = False

        from collections import OrderedDict
        classifier = nn.Sequential(OrderedDict([
            ('dropout', nn.Dropout(dropout)),
            ('inputs', nn.Linear(networks[network], hidden_layer1)),
            ('relu1', nn.ReLU()),
            ('hidden_layer1', nn.Linear(hidden_layer1, 90)),
            ('relu2', nn.ReLU()),
            ('hidden_layer2', nn.Linear(90, 80)),
            ('relu3', nn.ReLU()),
            ('hidden_layer3', nn.Linear(80, 102)),
            ('output', nn.LogSoftmax(dim=1))]))

        model.classifier = classifier
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.classifier.parameters(), lr)
        if torch.cuda.is_available() and power == 'gpu':
            model.cuda()
            
        return model, optimizer, criterion

def train_network(model, criterion, optimizer, dataloaders, dataloaders_validation, dataloaders_test, epochs = 3, print_every=10, power='gpu'):
    print("#### Training started ####")
    steps = 0
    loss_show = []
    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(dataloaders):
            steps += 1
            
            inputs, labels = inputs.to('cuda'), labels.to('cuda')

            optimizer.zero_grad()

            # FW and BW passes

            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                vlost = 0
                accuracy=0

                for ii, (inputs2,labels2) in enumerate(dataloaders_validation):
                    optimizer.zero_grad()
                    if torch.cuda.is_available():
                        inputs2, labels2 = inputs2.to('cuda:0') , labels2.to('cuda:0')
                        model.to('cuda:0')

                    with torch.no_grad():
                        outputs = model.forward(inputs2)
                        vlost = criterion(outputs,labels2)
                        ps = torch.exp(outputs).data
                        equality = (labels2.data == ps.max(1)[1])
                        accuracy += equality.type_as(torch.FloatTensor()).mean()

                vlost = vlost / len(dataloaders_validation)
                accuracy = accuracy /len(dataloaders_validation)

                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Loss: {:.4f}".format(running_loss/print_every),
                      "Validation Lost {:.4f}".format(vlost),
                       "Accuracy: {:.4f}".format(accuracy))

                running_loss = 0
    print("######")
    print("### Finished training ###")
    print("### Epochs: {} ###".format(epochs))
    print("### Steps: {} ###".format(steps))
    check_accuracy_on_test(dataloaders_test)
    print("######")


# TODO: Do validation on the test set
def check_accuracy_on_test(dataloaders_test):
    correct = 0
    total = 0
    if torch.cuda.is_available() and power == 'gpu':
        model.to('cuda:0')
    with torch.no_grad():
        for data in dataloaders_test:
            images, labels = data
            if torch.cuda.is_available() and power == 'gpu':
                images, labels = images.to('cuda'), labels.to('cuda')
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy on the test images: %d %%' % (100 * correct / total))


def save_checkpoint(path='checkpoint.pth',network ='densenet121', hidden_layer1=120,dropout=0.5,lr=0.001,epochs=4):

    model.class_to_idx = image_datasets.class_to_idx
    model.cpu
    torch.save({'network' :network,
                'hidden_layer1':120,
                'state_dict':model.state_dict(),
                'class_to_idx':model.class_to_idx},
                path)

def load_model(path):
    checkpoint=torch.load(path)
    network = checkpoint['network']
    hidden_layer1 = checkpoint['hidden_layer1']
    model,_,_ = nn_setup(network , 0.5,hidden_layer1)
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    pil = Image.open(image)
    adjust = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = adjust(pil)

    return img


def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax


def predict(image_path, model, topk=5, power='gpu'):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    if torch.cuda.is_available() and power=='gpu':
        model.to('cuda:0')
    image = process_image(image_path)
    image = image.unsqueeze_(0)
    image = image.float()

    if power == 'gpu':
        with torch.no_grad():
            output = model.forward(image.cuda())
    else:
        with torch.no_grad():
            output=model.forward(image)
    # Softmax = First dimension is your batch dimension, second is depth, third is rows and last one is columns.
    probability = F.softmax(output.data, dim=1)
    return probability.topk(topk)

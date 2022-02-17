# Load libraries
import os
import numpy as np
import torch
import glob
import torch.nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.autograd import Variable
import torchvision
import pathlib

# check which device you're using => cpu == slow || coda == fast
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Transforms
transformer = transforms.Compose([
    transforms.Resize((150,150)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), #0-255 to 0-1, numpy to tensors
    transforms.Normalize([0.5, 0.5, 0.5], # 0-1 to [-1,1], formula: (x-mean)/std (standard deviation)
                         [0.5, 0.5, 0.5])
    ])
# DataLoader

# Path for training and testing directory
train_path='C:/Users/ayoub/OneDrive/TMM/Stage fase 3/Arinti/FindWaldo/FindWaldo/Testing Area/scene_detection/seg_train'
test_path='C:/Users/ayoub/OneDrive/TMM/Stage fase 3/Arinti/FindWaldo/FindWaldo/Testing Area/scene_detection/seg_test'

train_loader = DataLoader(
    torchvision.datasets.ImageFolder(train_path,transform=transformer),
    batch_size=210, shuffle=True
)
test_loader = DataLoader(
    torchvision.datasets.ImageFolder(test_path,transform=transformer),
    batch_size=210, shuffle=True
)

# Categories
root=pathlib.Path(train_path)
classes=sorted([j.name.split('/')[-1] for j in root.iterdir()])
print(classes)

# CNN network
class ConvNet(nn.Module):
    def __init__(self,num_classes=6):
        super(ConvNet,self).__init__()
        
        # Ouput size after convution filter
        # ((w-f+2P)/2s) + 1
        
        # Input shape = (256,3,150,150) => 256 == batch_size || 3 == number of channels || 150,150 == height, width
        
        self.conv1=nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
        # Shape = (256, 12, 150, 150)
        self.bn1=nn.BatchNorm2d(num_features=12)
        # Shape = (256, 12, 150, 150)
        self.relu1=nn.ReLU()
        # Shape = (256, 12, 150, 150)
        
        self.pool=nn.MaxPool2d(kernel_size=2)
        # Reduce the image size by factor 2
        # Shape = (256, 12, 75, 75)
        
        self.conv2=nn.Conv2d(in_channels=12, out_channels=20, kernel_size=3, stride=1, padding=1)
        # Shape = (256, 20, 75, 75)
        self.relu2=nn.ReLU()
        # Shape = (256, 20, 75, 75)
        
        self.conv3=nn.Conv2d(in_channels=20, out_channels=32, kernel_size=3, stride=1, padding=1)
        # Shape = (256, 32, 75, 75)
        self.bn3=nn.BatchNorm2d(num_features=32)
        # Shape = (256, 32, 75, 75)
        self.relu3=nn.ReLU()
        # Shape = (256, 32, 75, 75)
        
        self.fc=nn.Linear(in_features=32*75*75, out_features=num_classes)

    # Feed forward function
    def forward(self, input):
        output=self.conv1(input)
        output=self.bn1(output)
        output=self.relu1(output)
            
        output=self.pool(output)
            
        output=self.conv2(output)
        output=self.relu2(output)
            
        output=self.conv3(output)
        output=self.bn3(output)
        output=self.relu3(output)
            
        # Above output will be in the matriix form, with shape (256, 32, 75, 75)
            
        output=output.view(-1,32*75*75)
            
        output=self.fc(output)
            
        return output

model = ConvNet(num_classes=6).to(device)
# Optimizer and loss function
optimizer=Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
loss_function=nn.CrossEntropyLoss()

num_epochs=6

# Calculating the size of training and testing images
train_count=len(glob.glob(train_path+'/**/*.jpg'))
test_count=len(glob.glob(test_path+'/**/*.jpg'))

print('Number of training pics: ',train_count)
print('Number of testing pics: ',test_count)

# Model training and saving best model
best_accuracy=0.0

for epoch in range(num_epochs):
    # Evaluation and training on training dataset
    model.train()
    train_accuracy = 0.0
    train_loss = 0.0
    print((len(train_loader)))
    for i, (image,labels) in enumerate(train_loader):
        images=Variable((image.cuda()))
        labels=Variable((labels.cuda()))
        
        
        optimizer.zero_grad()
        output = model(images)
        loss=loss_function(output,labels)
        loss.backward()
        optimizer.step()
        
        train_loss+= loss.cuda().data*images.size(0)
        _, prediction=torch.max(output.data,1)
        
        train_accuracy+=int(torch.sum(prediction==labels.data))
        train_accuracy+=int(torch.sum(prediction==labels.data))
        
    train_accuracy=train_accuracy/train_count
    train_loss=train_loss/train_count

    # Evaluation on testing dataset
    model.eval()
    
    test_accuracy=0.0
    for i, (images,labels) in enumerate(test_loader):
        images=Variable(image.cuda())
        labels=Variable(labels.cuda())
            
        outputs=model(images)
        _, prediction=torch.max(outputs.data,1)
        test_accuracy+=int(torch.sum(prediction==labels.data))
        
    test_accuracy=test_accuracy/test_count
    
    print("Epoch: " +str(epoch)+ " Train Loss: "+str(int(train_loss))+" Train Accuracy: " + str(train_accuracy)+" Test Accuracy: "+ str(test_accuracy))
    
    # Save the best model
    if test_accuracy>best_accuracy:
        torch.save(model.state_dict( ))

 
            
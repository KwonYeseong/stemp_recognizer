from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy
import skimage 
import os
import cv2
from torch.utils.data import Dataset
from torchvision import datasets, transforms

E = 0

class YeseongDataset(Dataset):
    def __init__(self, O_dir, X_dir, tem_dir, transform=None):
        self.O_dir = O_dir
        self.X_dir = X_dir
        self.transform = transform
        self.template_stamp = cv2.imread(tem_dir, cv2.IMREAD_GRAYSCALE)

    def __len__(self):
        temp = os.listdir(self.O_dir)
        O_list = [file for file in temp if file.endswith(".bmp")]
        temp = os.listdir(self.X_dir)
        X_list = [file for file in temp if file.endswith(".bmp")]

        return len(O_list) + len(X_list)

    def __getitem__(self, idx):
        temp = os.listdir(self.O_dir)
        O_list = [file for file in temp if file.endswith(".bmp")]
        temp = os.listdir(self.X_dir)
        X_list = [file for file in temp if file.endswith(".bmp")]

        image_list = O_list + X_list

        if idx < len(O_list):
            label = 1
            #image = skimage.io.imread(self.O_dir + image_list[idx])
            image = cv2.imread(self.O_dir + image_list[idx])
            
        elif idx < len(O_list) + len(X_list): 
            label = 0
            #image = skimage.io.imread(self.X_dir + image_list[idx])
            image = cv2.imread(self.X_dir + image_list[idx])

        image = self.rotateImage(image)

        # image = transforms.ToTensor()(image)
        image = self.transform(image)
        return [image, label]

    def rotateImage(self, image) :
        h, w, c = image.shape
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        temp = image
        degree = 15.0
        scale = 1.0
        min_temp = 1.0
        
        methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
            'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']
        

        for i in range(24):
            M = cv2.getRotationMatrix2D((w/2, h/2), degree * i, scale)
            rotatedImage = cv2.warpAffine(image, M, (w, h))

            res = cv2.matchTemplate(rotatedImage, self.template_stamp, eval('cv2.TM_CCORR'))
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

            if min_val > min_temp:
                temp = rotatedImage
                min_temp = min_val

        # cv2.imwrite("../rotated_images/" + str(self.index) + ".jpg", temp)
        return temp


    

# class LeNet(nn.Module):
#     def __init__(self):
#         super(LeNet, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=4, stride=1)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=2, stride=2)
#         self.fc1 = nn.Linear(4 * 4 * 64, 500)
#         self.fc2 = nn.Linear(500, 2)

#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.max_pool2d(x, 2, 2)
#         x = F.relu(self.conv2(x))
#         x = F.max_pool2d(x, 2, 2)
#         x = x.view(-1, 4 * 4 * 64)
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return F.log_softmax(x, dim=1)

# class LeNet(nn.Module):
#     def __init__(self):
#         super(LeNet, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=4, stride=2)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
#         self.conv3 = nn.Conv2d(64, 64, kernel_size=4, stride=2)
#         self.fc1 = nn.Linear(2 * 2 * 64, 500)
#         self.fc2 = nn.Linear(500, 2)

#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = F.relu(self.conv3(x))
#         x = x.view(-1, 2 * 2 * 64)
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return F.log_softmax(x, dim=1)

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        # self.bn1 = nn.BatchNorm2d(64)
        # self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=4, stride=2)
        self.fc1 = nn.Linear(2 * 2 * 64, 500)
        self.fc2 = nn.Linear(500, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # x = self.bn1(x)
        # x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = x.view(-1, 2 * 2 * 64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def train(args, model, device, train_data, optimizer, epoch):
    model.train()
    batch_idx = 0
    for (data, target) in train_data :
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        batch_idx = batch_idx + 1
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_data) * args.batch_size,
                100. * batch_idx / len(train_data), loss.item()))

def test(args, model, device, test_data):
    model.eval()
    test_loss = 0
    correct = 0
    cnt = 0
    with torch.no_grad():
        for data, target in test_data:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            check_error_image(output, target, data, cnt)
            cnt = cnt + 1
            
            test_loss += F.nll_loss(output, target).item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_data)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_data) * args.batch_size - 3,
        100. * correct / (len(test_data) * args.batch_size - 3)))

def valid(args, model, device, valid_data):
    model.eval()
    valid_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in valid_data:
            data, target = data.to(device), target.to(device)
            output = model(data)
            valid_loss += F.nll_loss(output, target).item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    valid_loss /= len(valid_data)

    print('\nValid set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        valid_loss, correct, len(valid_data) * args.batch_size - 2,
        100. * correct / (len(valid_data) * args.batch_size - 2)))

def make_dataloaders(dataset, train_size, valid_size, test_size, batch_size):
    indices = torch.randperm(len(dataset))
    train_indices = indices[:len(indices) - (valid_size+test_size)]
    valid_indices = indices[train_size:len(indices) - test_size]
    test_indices = indices[train_size + valid_size:]

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                sampler=torch.utils.data.SubsetRandomSampler(train_indices))
    valid_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                sampler=torch.utils.data.SubsetRandomSampler(valid_indices))                                            
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,  
                                                sampler=torch.utils.data.SubsetRandomSampler(test_indices))

    return train_loader, valid_loader, test_loader 

def check_error_image(results, labels, images, cnt):
    rFlag = 0
    lFlag = 0
    index = 0
    deNorm = transforms.Compose([
        transforms.Normalize(mean=(-0.5/0.5, ), std=(1/0.5,)),
        transforms.ToPILImage()
    ])
    
    for i in range(len(results)):
        if results[i][0] > results[i][1] : rFlag = 0
        else : rFlag = 1

        if labels[i].item() == 0 : lFlag = 0
        else : lFlag = 1
        # print("result {}, label {}".format(rFlag, lFlag))
        if rFlag != lFlag :
            #save image
            pilImage = deNorm(images[i])
            if lFlag == 1:
                pilImage.save("../error_images/" + str(E) + "_" + str(cnt) + "_" + str(index) + "_True"  + ".jpg")
            else :
                pilImage.save("../error_images/" + str(E) + "_" + str(cnt) + "_" + str(index) + "_False" + ".jpg")
            
            index = index + 1
    
    print("error count {} in {}th batch".format(index, cnt))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N', help='input batch size for training (default: 32')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N', help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,  help='For Saving the current Model')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    # prepare data
    transform = transforms.Compose([
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        # transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, ), std=(0.5,)),
    ])
    O_dir = '../O/'
    X_dir = '../X/'
    tem_dir = '../template_stamp.bmp'

    dset = YeseongDataset(O_dir, X_dir, tem_dir, transform)
    dLengh = [int(len(dset) * 0.7), int(len(dset) * 0.1), int(len(dset) * 0.2)]
    if sum(dLengh) != len(dset) :
        dLengh[0] = dLengh[0] + len(dset) - sum(dLengh)
    print("{} : {} : {} ".format(dLengh[0], dLengh[1], dLengh[2]))
    train_loader, val_loader, test_loader = make_dataloaders(dset, dLengh[0], dLengh[1], dLengh[2], args.batch_size)

    print("{} {} {} ".format(len(train_loader), len(val_loader), len(test_loader)))


    model = LeNet().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        global E
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)
        E += 1

    valid(args, model, device, val_loader)


    if (args.save_model):
        torch.save(model.state_dict(),"lenet_cnn.pt")
    
if __name__ == '__main__':
    main()
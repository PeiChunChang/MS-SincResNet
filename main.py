import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable
import scipy.io as sio
from models import *
from utils import progress_bar
import math
import csv
from dataloader import *

torch.multiprocessing.set_sharing_strategy('file_system')
torch.backends.cudnn.benchmark = True
VERSION = 0
EPOCH_MAX = 2
BATCH_SIZE = 8
CV_NUM = 10
INIT_LR = 0.005
CLASS_NUM = 10

torch.manual_seed(1) # cpu
torch.cuda.manual_seed(1) #gpu
np.random.seed(1) #numpy

def accuracy(predict, gt):
    _, pred_ans = torch.max(predict, 1)
    acc = torch.mean((pred_ans == gt).float())
    return acc

def accuracy_song(predict, gt):
    vote = np.zeros(CLASS_NUM)
    for i in predict:
        vote[i] += 1
    answer = np.argmax(vote, 0)
    return (1 if answer == gt[0] else 0)

def adjust_learning_rate(optimizer, epoch):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * 0.5

def warm_up(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(data, label):
    model.train()
    data, label = data.cuda(), label.cuda()
    output, _, _, _ = model(data)
    loss =  criteria(output, label)
    acc = accuracy(output, label)
    loss.backward()
    return loss, acc

def test(data, label):
    model.eval()
    with torch.no_grad():
        data, label = data.cuda(), label.cuda()
        output, _, _, _ = model(data)
        acc = accuracy(output, label)
        _, output = torch.max(output, 1)
    return acc, output

for cv in range(CV_NUM):
    TRAIN_PATH = '/media/maplepig/data11/GTZAN/fold_%d/train' % cv
    TEST_PATH = '/media/maplepig/data11/GTZAN/fold_%d/test' % cv
    OUTPUT_PATH = './v%d_output/fold_%d' % (VERSION, cv)
    if not os.path.exists(OUTPUT_PATH + '/checkpoints'):
        os.makedirs(OUTPUT_PATH + '/checkpoints')

    TrainDataLoader = torch.utils.data.DataLoader(
        myDataset_GTZAN(TRAIN_PATH, transform=True, loader=loader_GTZAN, train=True), 
        batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    TestDataLoader = torch.utils.data.DataLoader(
        myDataset_GTZAN(TEST_PATH, transform=False, loader=loader_GTZAN, train=False), 
        batch_size=1, shuffle=False, num_workers=8)

    model = MS_SincResNet()
    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
    model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=INIT_LR, momentum=0.9, weight_decay=1e-04, nesterov=True)
    optimizer.zero_grad()
    criteria = nn.CrossEntropyLoss()

    clip_acc = np.zeros(EPOCH_MAX)
    song_acc = np.zeros(EPOCH_MAX)
    train_acc = np.zeros(EPOCH_MAX)

    file = open(OUTPUT_PATH + '_record.csv', 'w', newline ='')
    header = ['epoch', 'train_loss', 'clip_acc', 'song_acc'] 
    writer = csv.DictWriter(file, fieldnames = header) 
    writer.writeheader()
    for epoch in range(EPOCH_MAX):
        if epoch < 5:
            warm_up(optimizer, 1e-05)
        if epoch == 5:
            warm_up(optimizer, INIT_LR)

        if epoch % 30 == 0 and epoch != 0:
            adjust_learning_rate(optimizer, epoch)
        
        total_train_loss = 0.0
        total_train_acc = 0.0
        for batch_idx, (data, label) in enumerate(TrainDataLoader):
            data = data.view(data.size()[0] * data.size()[1], data.size()[2], data.size()[3])
            label = label.view(label.size()[0] * label.size()[1])

            loss, acc = train(data, label)
            total_train_loss += loss
            total_train_acc += acc
            
            progress_bar(batch_idx, len(TrainDataLoader), 
            'Fold_%d Ep %d/%d avg. loss = %.4f acc = %.3f' %(cv, epoch, EPOCH_MAX, total_train_loss/(batch_idx+1), total_train_acc/(batch_idx+1)))
            optimizer.step()
            optimizer.zero_grad()


        savefilename = OUTPUT_PATH + '/checkpoints/checkpoint_' + str(epoch) + '.tar'
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'train_loss': total_train_loss/len(TrainDataLoader),
            'train_acc': total_train_acc/len(TrainDataLoader)
        }, savefilename)

        acc_V = 0.0
        s_acc = np.zeros(1)
        for batch_idx, (data, label) in enumerate(TestDataLoader):
            data.squeeze_(dim=0)
            label = label.squeeze(dim=0)
            acc, output = test(data, label)
            acc_V += acc
            s_acc += accuracy_song(output, label)
        clip_acc[epoch] = acc_V/len(TestDataLoader)
        song_acc[epoch] = s_acc/len(TestDataLoader)
        train_acc[epoch] = total_train_acc/len(TrainDataLoader)

        print('Testing: ACC: %.3f/%.3f.' %(acc_V/len(TestDataLoader),s_acc/len(TestDataLoader)))
        writer.writerow({
                'epoch': epoch,
                'train_loss': (total_train_loss/len(TrainDataLoader)).data.cpu().numpy(),
                'clip_acc': acc_V.cpu().numpy(),
                'song_acc': s_acc})
        optimizer.zero_grad()

    print('=======================================================')
    best_acc, best_epoch = np.max(song_acc), np.argmax(song_acc)
    print('The best acc = %f, (%d-th epoch)' %(best_acc, best_epoch))
    print('The stable acc =', np.mean(song_acc[175:200]))
    file_name = './v%d_output/fold_%d_CM.mat' % (VERSION, cv)
    sio.savemat(file_name, {"clip_acc": clip_acc, "song_acc": song_acc, "train_acc": train_acc})
    file.close()



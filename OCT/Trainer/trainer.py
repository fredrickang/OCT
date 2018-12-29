import time
import numpy as np
from sklearn.cross_validation import KFold

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable


def makefold(data, nfolds):
    nfold = nfolds
    kf = KFold(len(data), nfolds = nfold, shuffle = True)
    return kf


class Net(nn.Module):
    def __init__(self, num_classes=4):
        super(Net, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3),
            nn.MaxPool2d(2, 2),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3),
            nn.MaxPool2d(2, 2),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3),
            nn.MaxPool2d(2, 2),
            nn.ReLU(inplace=True)
        )

        self.classifier = nn.Sequential(
            nn.Linear(14 * 41 * 64, 200),
            nn.BatchNorm1d(200),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(200, 20),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(20, num_classes)
        )

    def forward(self, x):
        out = self.features(x)
        out = out.view(-1, 14 * 41 * 64)
        out = self.classifier(out)
        # out = F.softmax(out)
        if self.training:
            pass
        else:
            out = F.softmax(out)
        return out


def train(Model, train_x, train_y, nfolds, epochs=[150, 100, 50],
          batch_size=101, learning_rate=[0.01, 0.001, 0.0001], directory='./TorchModels/OCTmodel.pt'):
        '''
        train_x 는 np.array 여야 함. (/255. 작업을 마친.)
        train_y 도 마찬가지 (np.array() 거친.)'''

        num_fold = 0
        best_loss = 1000000  # value for saving the best model
        best_acc = 0.00

        kf_loss = np.zeros(nfolds)
        kf_accu = np.zeros(nfolds)

        for train_index, test_index in makefold(train_y, nfolds):
            start_time_kf = time.time()

            X_train = train_x[train_index]
            Y_train = train_y[train_index]
            X_valid = train_x[test_index]
            Y_valid = train_y[test_index]

            num_fold += 1
            print('Start KFold number {} from {}'.format(num_fold, nfolds))
            print('Split train: ', len(X_train), len(Y_train))
            print('Split valid: ', len(X_valid), len(Y_valid))

            batch_size = batch_size

            model = Model
            model.apply(weights_init)

            if torch.cuda.is_available():
                model.cuda()
                if torch.cuda.device_count() > 1:
                    print(" Use ", torch.cuda.device_count(), " GPUs.")
                    model = nn.DataParallel(model)

            best_loss_kf = 1000000  # value for saving the best model
            best_acc_kf = 0.00

            for lr, epo in zip(learning_rate, epochs):
                criterion = nn.CrossEntropyLoss().cuda()
                optimizer = optim.Adam(model.parameters(), lr=lr)
                # optimizer.zero_grad()

                for epoch in range(epo):

                    start_time_epoch = time.time()
                    num_suffle = np.random.permutation(range(len(Y_train)))

                    for step in range(int(len(X_train) / batch_size)):
                        model.zero_grad()

                        correct = 0
                        total = 0

                        x_batch = np.transpose(X_train[num_suffle[
                                                       step * batch_size:(step + 1) * batch_size]], (0, 3, 1, 2))
                        y_batch = Y_train[num_suffle[step * batch_size:(step + 1) * batch_size]]

                        input_var = Variable(torch.from_numpy(x_batch), requires_grad=True).float().cuda()
                        target_var = Variable(torch.from_numpy(y_batch)).cuda()
                        target_var = target_var.long()
                        # target_var = Variable(torch.LongTensor(y_batch)).cuda()

                        model.train()
                        output = model(input_var)
                        loss = criterion(output, target_var)
                        # optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        _, predicted = torch.max(output.data, 1)
                        predicted = predicted.cpu()
                        total += target_var.size(0)
                        target_var = target_var.cpu()
                        correct += (predicted.numpy() == target_var.data.numpy()).sum()
                        accuracy = 100 * correct / total

                        if step % 20 == 0:
                            valid_loss, valid_accu = validate(model, X_valid, Y_valid)

                            print("[Epochs: " + str(epoch + 1) + "  step: " + str(step) + " ]  / lr = " + str(lr))
                            # print(loss.data.cpu().numpy())
                            print("  Train loss: " + str(
                                loss.data.cpu().numpy().astype(float)) + " /  Train Accuracy: " + str(accuracy) + " %")
                            print("  Validation loss: " + str(valid_loss.data[0]) + " / Valid Accuracy: " + str(
                                valid_accu) + " %")

                            path = directory
                            if valid_loss.data[0] < best_loss:
                                best_loss = valid_loss.data[0]
                                best_acc = valid_accu
                                torch.save(model, path)
                                print("## The model saved in : " + path + '##')

                            if valid_loss.data[0] < best_loss_kf:
                                best_loss_kf = valid_loss.data[0]
                                best_acc_kf = valid_accu

                    elapsed_time_epoch = time.time() - start_time_epoch
                    print('End epoch number {} from {} / at {}s'.format(epoch + 1, epo, elapsed_time_epoch))
                    print(' ')

            kf_loss[num_fold - 1] = best_loss_kf
            kf_accu[num_fold - 1] = best_acc_kf

            # gc.collect()
            elapsed_time_kf = time.time() - start_time_kf
            print('End KFold number {} from {} / at {}s'.format(num_fold, nfolds, elapsed_time_kf))
            print('Best validation loss : {}  /  Best validation accuracy : {}%.'.format(kf_loss[num_fold - 1],
                                                                                         kf_accu[num_fold - 1]))
            print(' ')
            print('===================================================================================')
            print(' ')
            print(' ')

        # print(kf_accu)
        # print(type(kf_accu))
        print('Training is finished.')
        print('Model mean loss = {}  /  Model mean accuracy = {}%'.format(np.mean(kf_loss), np.mean(kf_accu)))


def validate(model_, valid_x, valid_y, batch_size=101, loss_fn=nn.CrossEntropyLoss()):
    criterion = loss_fn
    p_valid = []
    pred_true = []
    correct_v = 0
    total_v = 0
    for i in range(int(len(valid_x) / batch_size)):
        batch_x = np.transpose(valid_x[i * batch_size:(i + 1) * batch_size],
                               (0, 3, 1, 2))
        batch_y = valid_y[i * batch_size:(i + 1) * batch_size]

        input_val = Variable(torch.from_numpy(batch_x)).float().cuda()
        target_val = Variable(torch.LongTensor(batch_y)).cuda()

        model_.eval()
        output_val = model_(input_val)
        output_val = output_val.cpu().data.numpy().astype(float)

        p_valid.extend(output_val)
        pred_true.extend(batch_y)

    p_valid_v = Variable(torch.from_numpy(np.array(p_valid)), requires_grad=True).float()
    pred_true = Variable(torch.from_numpy(np.array(pred_true)))
    pred_true = pred_true.long()

    val_loss = criterion(p_valid_v, pred_true)

    _, predicted_v = torch.max(p_valid_v.data, 1)
    predicted_v = predicted_v.cpu()
    total_v += pred_true.size(0)
    pred_true = pred_true.cpu()
    correct_v += (predicted_v.numpy() == pred_true.data.numpy()).sum()
    accuracy = 100 * correct_v / total_v

    return val_loss, accuracy

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal(m.weight.data)
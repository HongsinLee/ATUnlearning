import time
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import torch
from torch import nn, optim
from tqdm import tqdm
from models import AllCNN
from cifar10_models import *
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

import torchattacks

def loss_picker(loss):
    if loss == 'mse':
        criterion = nn.MSELoss()
    elif loss == 'cross':
        criterion = nn.CrossEntropyLoss()
    else:
        print("automatically assign mse loss function to you...")
        criterion = nn.MSELoss()

    return criterion


def optimizer_picker(optimization, param, lr, momentum=0.):
    if optimization == 'adam':
        optimizer = optim.Adam(param, lr=lr)
    elif optimization == 'sgd':
        optimizer = optim.SGD(param, lr=lr, momentum=momentum)
    else:
        print("automatically assign adam optimization function to you...")
        optimizer = optim.Adam(param, lr=lr)
    return optimizer



def train(model, data_loader, criterion, optimizer, device='cpu'):
    running_loss = 0
    model.train()
    torchattackPGD = torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=10, random_start=True)
    for step, (batch_x, batch_y) in enumerate(tqdm(data_loader)):
        # print(batch_y.size())

        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        inputs_adv = torchattackPGD(batch_x, batch_y)
        optimizer.zero_grad()
        output = model(inputs_adv)  # get predict label of batch_x

        loss = criterion(output, batch_y)  # mse loss


        loss.backward()
        optimizer.step()
        running_loss += loss
    return running_loss


def train_save_model(train_loader, test_loader, model_name, num_epochs, ATmethod, device, path, args = None):
    start = time.time()

    num_classes = max(train_loader.dataset.targets) + 1  # if args.num_classes is None else args.num_classes
    if model_name == 'AllCNN':
        model = AllCNN(n_channels=3, num_classes=num_classes, filters_percentage=0.5).to(device)
    elif model_name == 'ResNet18':
        model = resnet18(num_classes=num_classes).to(device)


    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=2e-4)
    best_robust_acc = 0
    best_clean_acc = 0
    for epo in range(num_epochs):
        model.train()

        if epo == 0 or ATmethod == 'PGD':
            innerloss = torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=10, random_start=True)
        elif ATmethod == 'TRADES':
            innerloss = torchattacks.TPGD(model, eps=8/255, alpha=2/255, steps=10)
        else:
            raise NotImplementedError
        
        with tqdm(train_loader) as pbar:
            pbar.set_description(f'Epoch - {epo}')
            for step, (batch_x, batch_y) in enumerate(pbar):
                N,_,_,_ = batch_x.shape
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                inputs_adv = innerloss(batch_x, batch_y)
                optimizer.zero_grad()
                output = model(inputs_adv) 

                if epo == 0 or ATmethod == 'PGD':
                    criterion = nn.CrossEntropyLoss()
                    loss = criterion(output, batch_y)
                elif ATmethod == 'TRADES':
                    criterion_kl = nn.KLDivLoss(reduction='sum')
                    loss_natural = F.cross_entropy(model(batch_x), batch_y)
                    loss_robust = (1.0 / N) * criterion_kl(F.log_softmax(output, dim=1), F.softmax(model(batch_x), dim=1))
                    loss = loss_natural + 6 * loss_robust 
                else:
                    raise NotImplementedError
                
                pbar.set_postfix(loss = loss.item())

                loss.backward()
                optimizer.step()

        
        if epo in [100, 150]:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1




        if epo%1 == 0 :
            test_accs = []
            test_accs_adv = []
            model.eval()
            for step,(batch_x, batch_y) in enumerate(test_loader):
                torchattackPGD_eval = torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=20, random_start=True)
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                inputs_adv = torchattackPGD_eval(batch_x, batch_y)
                with torch.no_grad():
                    output = model(batch_x)
                    output_adv = model(inputs_adv)
            
                predictions_adv = np.argmax(output_adv.cpu().detach().numpy(),axis=1)
                predictions_adv = predictions_adv - batch_y.cpu().detach().numpy()
                
                predictions = np.argmax(output.cpu().detach().numpy(),axis=1)
                predictions = predictions - batch_y.cpu().detach().numpy()
                
                test_accs = test_accs + predictions.tolist()
                test_accs_adv = test_accs_adv + predictions_adv.tolist()
            test_accs = np.array(test_accs)
            test_accs_adv = np.array(test_accs_adv)
            test_acc = np.sum(test_accs==0)/len(test_accs)
            test_acc_adv = np.sum(test_accs_adv==0)/len(test_accs_adv)

            print('test acc:{}'.format(test_acc))
            print('PGD20 acc:{}'.format(test_acc_adv))
            if test_acc_adv >= best_robust_acc:
                best_robust_acc = test_acc_adv
                best_clean_acc = test_acc
                # state = {'net':model.state_dict(), 'features': feature_exct.state_dict()}
                torch.save(model, '{}.pth'.format(path))
    end = time.time()
    print('training time:', end-start, 's')
    return best_clean_acc, best_robust_acc


def test(model, loader):
    test_accs = []
    test_accs_adv = []
    model.eval()
    for step,(batch_x, batch_y) in enumerate(loader):
        torchattackPGD_eval = torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=20, random_start=True)
        inputs_adv = torchattackPGD_eval(batch_x, batch_y)
        with torch.no_grad():
            output = model(batch_x)
            output_adv = model(inputs_adv)
    
        predictions_adv = np.argmax(output_adv.cpu().detach().numpy(),axis=1)
        predictions_adv = predictions_adv - batch_y.cpu().detach().numpy()
        
        predictions = np.argmax(output.cpu().detach().numpy(),axis=1)
        predictions = predictions - batch_y.cpu().detach().numpy()
        
        test_accs = test_accs + predictions.tolist()
        test_accs_adv = test_accs_adv + predictions_adv.tolist()
    test_accs = np.array(test_accs)
    test_accs_adv = np.array(test_accs_adv)
    test_acc = np.sum(test_accs==0)/len(test_accs)
    test_acc_adv = np.sum(test_accs_adv==0)/len(test_accs_adv)

    print('test acc:{}'.format(test_acc))
    print('PGD20 acc:{}'.format(test_acc_adv))
    return test_acc, test_acc_adv


def eval(model, data_loader, mode='backdoor', print_perform=False, device='cpu', name=''):
    model.eval()  # switch to eval status

    y_true = []
    y_predict = []
    for step, (batch_x, batch_y) in enumerate(data_loader):

        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        batch_y_predict = model(batch_x)
        if mode == 'pruned':
            batch_y_predict = batch_y_predict[:, 0:10]

        batch_y_predict = torch.argmax(batch_y_predict, dim=1)
        # batch_y = torch.argmax(batch_y, dim=1)
        y_predict.append(batch_y_predict)
        y_true.append(batch_y)

    y_true = torch.cat(y_true, 0)
    y_predict = torch.cat(y_predict, 0)

    num_hits = (y_true == y_predict).float().sum()
    acc = num_hits / y_true.shape[0]
    # print()

    if print_perform and mode != 'backdoor' and mode != 'widen' and mode != 'pruned':
        print(classification_report(y_true.cpu(), y_predict.cpu(), target_names=data_loader.dataset.classes, digits=4))
    if print_perform and mode == 'widen':
        class_name = data_loader.dataset.classes.append('extra class')
        print(classification_report(y_true.cpu(), y_predict.cpu(), target_names=class_name, digits=4))
        C = confusion_matrix(y_true.cpu(), y_predict.cpu(), labels=class_name)
        plt.matshow(C, cmap=plt.cm.Reds)
        plt.ylabel('True Label')
        plt.xlabel('Pred Label')
        plt.show()
    if print_perform and mode == 'pruned':
        # print(classification_report(y_true.cpu(), y_predict.cpu(), target_names=data_loader.dataset.classes, digits=4))
        class_name = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]#['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        C = confusion_matrix(y_true.cpu(), y_predict.cpu(), labels=class_name)
        plt.matshow(C, cmap=plt.cm.Reds)
        plt.ylabel('True Label')
        plt.xlabel('Pred Label')
        plt.title('{} confusion matrix'.format(name), loc='center')
        plt.show()

    return accuracy_score(y_true.cpu(), y_predict.cpu()), acc


def robust_eval(model, data_loader, mode='backdoor', device='cpu'):
    model.eval()  # switch to eval status
    torchattackPGD_eval = torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=20, random_start=True)
    y_true = []
    y_predict = []
    y_predict_adv = []
    for step, (batch_x, batch_y) in enumerate(data_loader):

        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        inputs_adv = torchattackPGD_eval(batch_x, batch_y)
        batch_y_predict = model(batch_x)
        batch_y_predict_adv = model(inputs_adv)
        if mode == 'pruned':
            batch_y_predict = batch_y_predict[:, 0:10]
            batch_y_predict_adv = batch_y_predict_adv[:, 0:10]
        batch_y_predict = torch.argmax(batch_y_predict, dim=1)
        batch_y_predict_adv = torch.argmax(batch_y_predict_adv, dim=1)
        # batch_y = torch.argmax(batch_y, dim=1)
        y_predict.append(batch_y_predict)
        y_predict_adv.append(batch_y_predict_adv)
        y_true.append(batch_y)

    y_true = torch.cat(y_true, 0)
    y_predict = torch.cat(y_predict, 0)
    y_predict_adv = torch.cat(y_predict_adv, 0)

    num_hits = (y_true == y_predict).float().sum()
    num_hits_adv = (y_true == y_predict_adv).float().sum()

    acc = num_hits / y_true.shape[0]
    acc_adv = num_hits_adv / y_true.shape[0]

    return acc, acc_adv


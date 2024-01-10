import copy
import utils
from trainer import eval, loss_picker, optimizer_picker, robust_eval
import numpy as np
import torch
from torch import nn
from adv_generator import LinfPGD, inf_generator, FGSM
import tqdm
import time
from models import init_params as w_init
from expand_exp import curvature, weight_assign
import torchattacks
import torch.nn.functional as F

def robust_evaluate(model, data_loader, device):
    test_accs = []
    test_accs_adv = []
    model.eval()
    for step,(batch_x, batch_y) in enumerate(data_loader):
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

    return test_acc, test_acc_adv


def AT_boundary_shrink(ori_model, train_forget_loader, dt, dv, test_loader, device, evaluate = True,
                    bound=0.1, poison_epoch=10, forget_class=0, unlearn_innerloss='FGSM', unlearn_ATmethod='PGD', path='./'):
    

    test_model = copy.deepcopy(ori_model).to(device)
    unlearn_model = copy.deepcopy(ori_model).to(device)
    start_time = time.time()

    if unlearn_innerloss == 'FGSM':
        innerloss = torchattacks.FGSM(test_model, eps=bound)
    elif unlearn_innerloss == 'PGD':
        innerloss = torchattacks.PGD(test_model, eps=bound, alpha=bound/4, steps=10)


    if unlearn_ATmethod == 'PGD':
        ATinner = torchattacks.PGD(test_model, eps=8/255, alpha=2/255, steps=10, random_start=True)
    elif unlearn_ATmethod == 'TRADES':
        ATinner = torchattacks.TPGD(test_model, eps=8/255, alpha=2/255, steps=10, random_start=True)
    elif unlearn_ATmethod == 'Nat':
        ATinner = torchattacks.VANILA(test_model)
    else:
        raise NotImplementedError

    forget_data_gen = inf_generator(train_forget_loader)
    batches_per_epoch = len(train_forget_loader)

    optimizer = torch.optim.SGD(unlearn_model.parameters(), lr=0.00001, momentum=0.9)

    num_hits = 0
    num_sum = 0
    nearest_label = []

    for itr in tqdm.tqdm(range(poison_epoch * batches_per_epoch)):

        x, y = forget_data_gen.__next__()
        N,_,_,_ = x.shape
        x = x.to(device)
        y = y.to(device)
        test_model.eval()
        x_adv = innerloss(x, y)
        adv_logits = test_model(x_adv)
        pred_label = torch.argmax(adv_logits, dim=1)
        if itr >= (poison_epoch - 1) * batches_per_epoch:
            nearest_label.append(pred_label.tolist())
        num_hits += (y != pred_label).float().sum()
        num_sum += y.shape[0]

        # adv_train
        unlearn_model.train()
        unlearn_model.zero_grad()
        optimizer.zero_grad()

        x_adv_new = ATinner(x, pred_label)
        ori_logits = unlearn_model(x_adv_new)

        if unlearn_ATmethod == 'PGD':
            criterion = nn.CrossEntropyLoss()
            loss = criterion(ori_logits, pred_label)
        elif unlearn_ATmethod == 'Nat':
            criterion = nn.CrossEntropyLoss()
            loss = criterion(ori_logits, pred_label)
        elif unlearn_ATmethod == 'TRADES':
            criterion_kl = nn.KLDivLoss(reduction='sum')
            loss_natural = F.cross_entropy(unlearn_model(x), pred_label)
            loss_robust = (1.0 / N) * criterion_kl(F.log_softmax(ori_logits, dim=1), F.softmax(unlearn_model(x), dim=1))
            loss = loss_natural + 6 * loss_robust 
        else:
            raise NotImplementedError


        
        loss.backward()
        optimizer.step()

    print('attack success ratio:', (num_hits / num_sum).float())
    print('boundary shrink time:', (time.time() - start_time))
    torch.save(unlearn_model, '{}boundary_shrink_unlearn_model.pth'.format(path))

    test_forget_loader, test_remain_loader = utils.get_forget_loader(dv, forget_class)
    _, train_remain_loader = utils.get_forget_loader(dt, forget_class)

    mode = 'pruned' if evaluate else ''
    test_acc, test_acc_adv = robust_eval(model=unlearn_model, data_loader=test_loader, mode=mode, print_perform=evaluate, device=device,
                       name='test set all class')
    forget_acc, forget_acc_adv = robust_eval(model=unlearn_model, data_loader=test_forget_loader, mode=mode, print_perform=evaluate,
                         device=device, name='test set forget class')
    remain_acc, remain_acc_adv = robust_eval(model=unlearn_model, data_loader=test_remain_loader, mode=mode, print_perform=evaluate,
                         device=device, name='test set remain class')
    train_forget_acc, train_forget_acc_adv = robust_eval(model=unlearn_model, data_loader=train_forget_loader, mode=mode, print_perform=evaluate,
                               device=device, name='train set forget class')
    train_remain_acc, train_remain_acc_adv = robust_eval(model=unlearn_model, data_loader=train_remain_loader, mode=mode, print_perform=evaluate,
                               device=device, name='train set remain class')
    print('test acc:{:.2%}, forget acc:{:.2%}, remain acc:{:.2%}, train forget acc:{:.2%}, train remain acc:{:.2%}\n'
          .format(test_acc, forget_acc, remain_acc, train_forget_acc, train_remain_acc))
    print('test acc adv:{:.2%}, forget acc adv:{:.2%}, remain acc adv:{:.2%}, train forget acc adv:{:.2%}, train remain acc adv:{:.2%}'
          .format(test_acc_adv, forget_acc_adv, remain_acc_adv, train_forget_acc_adv, train_remain_acc_adv))
    return unlearn_model



def boundary_shrink(ori_model, train_forget_loader, dt, dv, test_loader, device, evaluate = True,
                    bound=0.1, poison_epoch=10, forget_class=0, unlearn_innerloss='FGSM', unlearn_ATmethod='PGD', path='./'):
    

    test_model = copy.deepcopy(ori_model).to(device)
    unlearn_model = copy.deepcopy(ori_model).to(device)
    start_time = time.time()

    if unlearn_innerloss == 'FGSM':
        innerloss = torchattacks.FGSM(test_model, eps=bound)
    elif unlearn_innerloss == 'PGD':
        innerloss = torchattacks.PGD(test_model, eps=bound, alpha=bound/4, steps=10)

    forget_data_gen = inf_generator(train_forget_loader)
    batches_per_epoch = len(train_forget_loader)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(unlearn_model.parameters(), lr=0.00001, momentum=0.9)

    num_hits = 0
    num_sum = 0
    nearest_label = []

    for itr in tqdm.tqdm(range(poison_epoch * batches_per_epoch)):

        x, y = forget_data_gen.__next__()
        x = x.to(device)
        y = y.to(device)
        test_model.eval()
        x_adv = innerloss(x, y)
        adv_logits = test_model(x_adv)
        pred_label = torch.argmax(adv_logits, dim=1)
        if itr >= (poison_epoch - 1) * batches_per_epoch:
            nearest_label.append(pred_label.tolist())
        num_hits += (y != pred_label).float().sum()
        num_sum += y.shape[0]

        # adv_train
        unlearn_model.train()
        unlearn_model.zero_grad()
        optimizer.zero_grad()

        ori_logits = unlearn_model(x)
        ori_loss = criterion(ori_logits, pred_label)

        loss = ori_loss 
        
        loss.backward()
        optimizer.step()

    print('attack success ratio:', (num_hits / num_sum).float())
    print('boundary shrink time:', (time.time() - start_time))
    torch.save(unlearn_model, '{}boundary_shrink_unlearn_model.pth'.format(path))

    test_forget_loader, test_remain_loader = utils.get_forget_loader(dv, forget_class)
    _, train_remain_loader = utils.get_forget_loader(dt, forget_class)

    mode = 'pruned' if evaluate else ''
    _, test_acc = eval(model=unlearn_model, data_loader=test_loader, mode=mode, print_perform=evaluate, device=device,
                       name='test set all class')
    _, forget_acc = eval(model=unlearn_model, data_loader=test_forget_loader, mode=mode, print_perform=evaluate,
                         device=device, name='test set forget class')
    _, remain_acc = eval(model=unlearn_model, data_loader=test_remain_loader, mode=mode, print_perform=evaluate,
                         device=device, name='test set remain class')
    _, train_forget_acc = eval(model=unlearn_model, data_loader=train_forget_loader, mode=mode, print_perform=evaluate,
                               device=device, name='train set forget class')
    _, train_remain_acc = eval(model=unlearn_model, data_loader=train_remain_loader, mode=mode, print_perform=evaluate,
                               device=device, name='train set remain class')
    print('test acc:{:.2%}, forget acc:{:.2%}, remain acc:{:.2%}, train forget acc:{:.2%}, train remain acc:{:.2%}'
          .format(test_acc, forget_acc, remain_acc, train_forget_acc, train_remain_acc))

    return unlearn_model

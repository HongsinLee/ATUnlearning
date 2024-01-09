import argparse
import numpy as np
import boundary_unlearning
from utils import *
from trainer import *
from autoattack import AutoAttack


def seed_torch(seed=2022):
    np.random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    seed_torch()
    parser = argparse.ArgumentParser("Boundary Unlearning")
    parser.add_argument('--method', type=str, default='boundary_shrink',
                        choices=['boundary_shrink', 'boundary_expanding'], help='unlearning method')
    parser.add_argument('--ATmethod', type=str, default='PGD', help='unlearning method')
    parser.add_argument('--unlearn_innerloss', type=str, default='FGSM', help='unlearning method')
    parser.add_argument('--unlearn_ATmethod', type=str, default='PGD', help='unlearning method')
    parser.add_argument('--data_name', type=str, default='cifar10', choices=['mnist', 'cifar10'],
                        help='dataset, mnist or cifar10')
    parser.add_argument('--model_name', type=str, default='ResNet18', choices=['MNISTNet', 'AllCNN', 'ResNet18'], help='model name')
    parser.add_argument('--optim_name', type=str, default='sgd', choices=['sgd', 'adam'], help='optimizer name')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--epoch', type=int, default=200, help='training epoch')
    parser.add_argument('--forget_class', type=int, default=4, help='forget class')
    parser.add_argument('--dataset_dir', type=str, default='/mnt/server7_hard3/hongsin/dataset', help='dataset directory')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='checkpoints directory')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--train_or_unlearning', type=str, help='Train model from scratch')
    parser.add_argument('--evaluation', action='store_true', help='evaluate unlearn model')
    parser.add_argument('--extra_exp', type=str, help='optional extra experiment for boundary shrink',
                        choices=['curv', 'weight_assign', None])
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    create_dir(args.checkpoint_dir)
    create_dir(args.checkpoint_dir + '/' + args.data_name)
    create_dir(args.checkpoint_dir + '/' + args.data_name + '/' + args.ATmethod)
    path = args.checkpoint_dir + '/' + args.data_name + '/' + args.ATmethod + '/'

    
    trainset, testset = get_dataset(args.data_name, args.dataset_dir)
    train_loader, test_loader = get_dataloader(trainset, testset, args.batch_size, device=device)

    forget_class = args.forget_class
    num_forget = 5000
    train_forget_loader, train_remain_loader, test_forget_loader, test_remain_loader, repair_class_loader, \
    train_forget_index, train_remain_index, test_forget_index, test_remain_index \
        = get_unlearn_loader(trainset, testset, forget_class, args.batch_size, num_forget)

    if args.train_or_unlearning == "train":
        print('=' * 100)
        print(' ' * 25 + 'train original model from scratch')
        print('=' * 100)
        ori_model, ori_clean_acc, ori_robust_acc = train_save_model(train_loader, test_loader, args.model_name,
                                     args.epoch, args.ATmethod, device, path + "original_model")
        ori_model.eval()
        autoattack = AutoAttack(ori_model, norm='Linf', eps=8/255.0, version='standard')

        x_total = [x for (x, y) in test_loader]
        y_total = [y for (x, y) in test_loader]
        x_total = torch.cat(x_total, 0)
        y_total = torch.cat(y_total, 0)
        _, AA_acc = autoattack.run_standard_evaluation(x_total, y_total)

        print('\noriginal model acc:{:.4f}, PGD acc: {:.4f}, AutoAttack acc: {}'.format(ori_clean_acc, ori_robust_acc, AA_acc))
        file = open(path + "accuracy_summary.txt", 'a')
        file.write('original model \t\t\t\t\t Clean acc:{:.4f}, PGD acc: {:.4f}, AutoAttack acc: {}\n'.format(ori_clean_acc, ori_robust_acc, AA_acc))
        file.close()
    elif args.train_or_unlearning == "retrain":
        print('=' * 100)
        print(' ' * 25 + 'train retrained model from scratch')
        print('=' * 100)
        retrain_model, retrain_clean_acc, retrain_robust_acc = train_save_model(train_remain_loader, test_remain_loader, args.model_name,
                                         args.epoch, args.ATmethod, device, path + "forget" + str(args.forget_class) +"_retrain_model")
        
        retrain_model.eval()
        autoattack = AutoAttack(retrain_model, norm='Linf', eps=8/255.0, version='standard')

        x_total = [x for (x, y) in test_loader]
        y_total = [y for (x, y) in test_loader]
        x_total = torch.cat(x_total, 0)
        y_total = torch.cat(y_total, 0)
        _, AA_acc = autoattack.run_standard_evaluation(x_total, y_total)

        print('\nretrain model(forgetting :{}) acc:{:.4f}, PGD acc: {:.4f}, AutoAttack acc: {}'.format(args.forget_class, retrain_clean_acc, retrain_robust_acc, AA_acc))
        file = open(path + "accuracy_summary.txt", 'a')
        file.write('retrain model(forgetting :{})\t Clean acc:{:.4f}, PGD acc: {:.4f}, AutoAttack acc: {}\n'.format(args.forget_class, retrain_clean_acc, retrain_robust_acc, AA_acc))
        file.close()

    elif args.train_or_unlearning == "unlearning":
        print('=' * 100)
        print(' ' * 25 + 'load original model and retrain model')
        print('=' * 100)
        ori_model = torch.load('{}.pth'.format(path + "original_model"),
                               map_location=torch.device('cpu')).to(device)
        retrain_model = torch.load('{}.pth'.format(
            path + "forget" + str(args.forget_class) +"_retrain_model"), map_location=torch.device('cpu')).to(device)
        


        if args.method == 'boundary_shrink':
            print('*' * 100)
            print(' ' * 25 + 'begin boundary shrink unlearning')
            if args.extra_exp:
                print(' ' * 20 + 'with extra experiment curvature regularization' if args.extra_exp == 'curv' else
                    ' ' * 20 + 'with extra experiment weight assign')
            print('*' * 100)
            unlearn_model = boundary_unlearning.boundary_shrink(ori_model, train_forget_loader, trainset, testset,
                                                                test_loader, device, args.evaluation,
                                                                forget_class=args.forget_class, path=path,
                                                                extra_exp=args.extra_exp)
        elif args.method == 'boundary_expanding':
            print('*' * 100)
            print(' ' * 25 + 'begin boundary expanding unlearning')
            print('*' * 100)
            unlearn_model = boundary_unlearning.boundary_expanding(ori_model, train_forget_loader, test_loader,
                                                                test_forget_loader, test_remain_loader,
                                                                train_remain_loader, args.optim_name, device,
                                                                args.evaluation, path=path)

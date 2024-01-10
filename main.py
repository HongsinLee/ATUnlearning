import argparse
import numpy as np
import boundary_unlearning
import AT_unlearning
from utils import *
from trainer import *
from autoattack import AutoAttack
import wandb
import pdb


def seed_torch(seed=2022):
    np.random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    seed_torch()
    args = load_parser()
    args = load_config(args)
    print(args)
    
    if args.wand:
        assert wandb is not None, "Wandb not installed, please install it or run without wandb"
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args), name=args.wandb_name, tags=[args.wandb_tags])
        args.wandb_url = wandb.run.get_url()
        wandb.save(args.base_dir+'/config/' + args.config)
        args.wand_log = wandb
    
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
        ori_clean_acc, ori_robust_acc = train_save_model(train_loader, test_loader, args.model_name,
                                     args.epoch, args.ATmethod, device, path + "original_model", args)
        
        ori_model = torch.load('{}.pth'.format(path + "original_model"),
                        map_location=torch.device('cpu')).to(device)
        
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
        retrain_clean_acc, retrain_robust_acc = train_save_model(train_remain_loader, test_remain_loader, args.model_name,
                                         args.epoch, args.ATmethod, device, path + "forget" + str(args.forget_class) +"_retrain_model", args)
        
        retrain_model = torch.load('{}.pth'.format(
            path + "forget" + str(args.forget_class) +"_retrain_model"), map_location=torch.device('cpu')).to(device)
        retrain_model.eval()
        autoattack = AutoAttack(retrain_model, norm='Linf', eps=8/255.0, version='standard')

        x_total = [x for (x, y) in test_remain_loader]
        y_total = [y for (x, y) in test_remain_loader]
        x_total = torch.cat(x_total, 0)
        y_total = torch.cat(y_total, 0)
        _, AA_acc = autoattack.run_standard_evaluation(x_total, y_total)

        test_acc, test_acc_adv = robust_eval(model=retrain_model, data_loader=test_loader, device=device)
        forget_acc, forget_acc_adv = robust_eval(model=retrain_model, data_loader=test_forget_loader,device=device)
        remain_acc, remain_acc_adv = robust_eval(model=retrain_model, data_loader=test_remain_loader,device=device)
        train_forget_acc, train_forget_acc_adv = robust_eval(model=retrain_model, data_loader=train_forget_loader, device=device)
        train_remain_acc, train_remain_acc_adv = robust_eval(model=retrain_model, data_loader=train_remain_loader, device=device)

        print('test acc:{:.2%}, forget acc:{:.2%}, remain acc:{:.2%}, train forget acc:{:.2%}, train remain acc:{:.2%}\n'
            .format(test_acc, forget_acc, remain_acc, train_forget_acc, train_remain_acc))
        print('test acc adv:{:.2%}, forget acc adv:{:.2%}, remain acc adv:{:.2%}, train forget acc adv:{:.2%}, train remain acc adv:{:.2%}'
            .format(test_acc_adv, forget_acc_adv, remain_acc_adv, train_forget_acc_adv, train_remain_acc_adv))


        print('\nretrain model(forgetting :{}) acc:{:.4f}, PGD acc: {:.4f}, AutoAttack acc: {:.4f}'.format(args.forget_class, retrain_clean_acc, retrain_robust_acc, AA_acc))
        file = open(path + "accuracy_summary.txt", 'a')
        file.write('retrain model(forgetting :{})\t Clean acc:{:.4f}, PGD acc: {:.4f}, AutoAttack acc: {:.4f}\n'.format(args.forget_class, retrain_clean_acc, retrain_robust_acc, AA_acc))
        file.write('\t\t\t\t\t\t\t\t test acc:{:.2%}, forget acc:{:.2%}, remain acc:{:.2%}, train forget acc:{:.2%}, train remain acc:{:.2%}\n'
            .format(test_acc, forget_acc, remain_acc, train_forget_acc, train_remain_acc))
        file.write('\t\t\t\t\t\t\t\t test acc adv:{:.2%}, forget acc adv:{:.2%}, remain acc adv:{:.2%}, train forget acc adv:{:.2%}, train remain acc adv:{:.2%}'
            .format(test_acc_adv, forget_acc_adv, remain_acc_adv, train_forget_acc_adv, train_remain_acc_adv))
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
                                                                extra_exp=args.extra_exp, args = args)
        elif args.method == 'boundary_expanding':
            print('*' * 100)
            print(' ' * 25 + 'begin boundary expanding unlearning')
            print('*' * 100)
            unlearn_model = boundary_unlearning.boundary_expanding(ori_model, train_forget_loader, test_loader,
                                                                test_forget_loader, test_remain_loader,
                                                                train_remain_loader, args.optim_name, device,
                                                                args.evaluation, path=path, args = args)
            
        elif args.method == 'AT_BS':
            print('*' * 100)
            print(' ' * 25 + 'begin AT boundary shrink unlearning')
            print('*' * 100)
            unlearn_model = AT_unlearning.AT_boundary_shrink(ori_model, train_forget_loader, trainset, testset,
                                                                test_loader, device, forget_class=args.forget_class, 
                                                                unlearn_innerloss= args.unlearn_innerloss, unlearn_ATmethod = args.unlearn_ATmethod, path=path, args = args)
            
        elif args.method == 'BS':
            print('*' * 100)
            print(' ' * 25 + 'begin boundary shrink unlearning')
            print('*' * 100)
            unlearn_model = AT_unlearning.boundary_shrink(ori_model, train_forget_loader, trainset, testset,
                                                                test_loader, device, forget_class=args.forget_class, 
                                                                unlearn_innerloss= args.unlearn_innerloss, unlearn_ATmethod = args.unlearn_ATmethod, path=path, args = args)
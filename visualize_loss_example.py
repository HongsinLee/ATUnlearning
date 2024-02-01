import torch
import torchattacks
from visualize_land import *  
import os

def visualize(model, dataloader):
    
    eps = 16
    attack_FGSM = torchattacks.FGSM(model,eps = eps/255)
    attack_PGD = torchattacks.PGD(model,eps = eps/255 ,steps= 1, alpha = (eps/255))

    for i, data in enumerate(dataloader):
        x, y, _ = data
    
        rademacher_vec = 2.*(torch.randint(2, size=x.shape)-1.) * eps/255


        x = x.to('cuda')
        y = y.long().to('cuda')
        
        png_path = lambda x: os.path.join("./visualize/", 'task_idx_{}_{}_{}.png'.format(model_name,model_1_task,y[0].item()))
    
        adv_vec = attack_FGSM(x, y) - x
        adv_vec = adv_vec[0]
        rademacher_vec = 2.*(torch.randint(2, size=adv_vec.shape)-1.) * eps/255
        rademacher_vec2 = 2.*(torch.randint(2, size=adv_vec.shape)-1.) * eps/255
        
        adv_vec_1_2 = attack_PGD(x, y) - x
        adv_vec_1_2 = adv_vec_1_2[0]
        
        x_ = x[0]
        y_ = y[0]

        rx, ry, zs = compute_perturb(model=model,
                                image=x_, label=y_,
                                vec_x=adv_vec, vec_y=adv_vec_1_2,
                                range_x=(-1,1), range_y=(-1,1),
                                grid_size=50,
                                loss=nn.CrossEntropyLoss(reduction='none'),
                                batch_size = 25)
        
    
        print('computed adversarial loss landscape')
        plot_perturb_plt(rx, ry, zs, png_path, eps, xlabel='Adv1', ylabel='Adv2',)
        
   
    

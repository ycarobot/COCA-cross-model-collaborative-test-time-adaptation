"""
Copyright to SAR Authors, ICLR 2023 Oral (notable-top-5%)
built upon on Tent and EATA code.
"""
from logging import debug
import os
import time
import argparse
import random
import numpy as np
# from pycm import *
import math
from dataset.selectedRotateImageFolder import prepare_test_data
import torch    
import torch.nn.functional as F
import tent
import sar
from sam import SAM
import timm
import matplotlib.pyplot as plt
import numpy as np
from dataset.ImageNetMask import imagenet_r_mask
from dataset.ImagenetaMask import imagenet_a_mask
import models.Res as Resnet
import ssl
import matplotlib.pyplot as plt
from utils.utils import get_logger
from utils.cli_utils import *
from safetensors.torch import load_file
import sam
import eataorg
import pandas as pd
import deyo,roid
import COCAV4

import cotta

def validate(val_loader, model, criterion, args, mode='eval'):
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1, top5],
        prefix='Test: ')
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, dl in enumerate(val_loader):
            images, target = dl[0], dl[1]
            if args.gpu is not None:
                images = images.to(args.gpu)
            if torch.cuda.is_available():
                target = target.to(args.gpu)
            # compute output
            output = model(images,return_feature=False)
            # _, targets = output.max(1)
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % args.print_freq == 0:
                progress.display(i)
            if i > 10 and args.debug:
                break
    return top1.avg, top5.avg

def calfisher(net,args,fisher_loader,cnn=False):
    cnnnet = eata.configure_model(net)
    params, param_names = eata.collect_params(net)
    # fishers = None
    ewc_optimizer = torch.optim.SGD(params, 0.001)
    cnnfishers = {}
    train_loss_fn = nn.CrossEntropyLoss().cuda()
    for iter_, (images,images_transform, targets) in enumerate(fisher_loader, start=1):      
        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
            images_transform = images_transform.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            targets = targets.cuda(args.gpu, non_blocking=True)
            if cnn:
                outputs = cnnnet(images)
            else:
                outputs = cnnnet(images_transform)
        _, targets = outputs.max(1)
        loss = train_loss_fn(outputs, targets)
        loss.backward()
        for name, param in cnnnet.named_parameters():
            if param.grad is not None:
                if iter_ > 1:
                    cnnfisher = param.grad.data.clone().detach() ** 2 + cnnfishers[name][0]
                else:
                    cnnfisher = param.grad.data.clone().detach() ** 2
                if iter_ == len(fisher_loader):
                    cnnfisher = cnnfisher / iter_
                cnnfishers.update({name: [cnnfisher, param.data.clone().detach()]})
        ewc_optimizer.zero_grad()
    logger.info("compute netfisher matrices finished")
    del ewc_optimizer
    return cnnfishers



@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

def get_args():

    parser = argparse.ArgumentParser(description='MUTTA exps')
    # path
    parser.add_argument('--data', default='ImageNet_val', help='path to dataset')
    parser.add_argument('--data_corruption', default='ImageNet-C', help='path to corruption dataset')
    parser.add_argument('--data_ia', default='imagenet-a', help='path to dataset')
    parser.add_argument('--data_ir', default='ImageNet_r', help='path to dataset')
    parser.add_argument('--data_sketch', default='sketch', help='path to dataset')
    parser.add_argument('--dset', default='ic', help='dataset')
    parser.add_argument('--output', default='./cocaoutputs', help='the output directory of this experiment')
    parser.add_argument('--seed', default=2021, type=int, help='seed for initializing training. ')
    parser.add_argument('--gpu', default=6,type=int, help='GPU id to use.')
    parser.add_argument('--gpu2', default=6, type=int, help='GPU id to use.')
    parser.add_argument('--debug', default=False, type=bool, help='debug or not.')
    # dataloader
    parser.add_argument('--workers', default=4, type=int, help='number of data loading workers (default: 4)')
    parser.add_argument('--test_batch_size', default=64, type=int, help='mini-batch size for testing, before default value is 4')
    parser.add_argument('--if_shuffle', default=True, type=bool, help='if shuffle the test set.')
    # corruption settings
    parser.add_argument('--level', default=5, type=int, help='corruption level of test(val) set.')
    parser.add_argument('--corruption', default='gaussian_noise', type=str, help='corruption type of test(val) set.')
    # eata settings
    parser.add_argument('--fisher_size', default=2000, type=int, help='number of samples to compute fisher information matrix.')
    parser.add_argument('--fisher_alpha', type=float, default=2000., help='the trade-off between entropy and regularization loss, in Eqn. (8)')
    parser.add_argument('--e_margin', type=float, default=math.log(1000)*0.40, help='entropy margin E_0 in Eqn. (3) for filtering reliable samples')
    parser.add_argument('--d_margin', type=float, default=0.05, help='\epsilon in Eqn. (5) for filtering redundant samples')
    # MUTTA framework Settings
    parser.add_argument('--method', default='coca', type=str, help='noadapt,baseline,coca,cotta')
    parser.add_argument('--plusmethod', default='tent', type=str, help='tent, eata,sar')
    parser.add_argument('--model1', default='Resnet50', type=str, help='mobilevit,Resnet18,efficientvit,vitbase,vitlarge,Resnet50,Resnet101')
    parser.add_argument('--model2', default='vitbase', type=str, help='mobilevit,Resnet18,Resnet18-2,efficientvit,vitbase,vitlarge,Resnet50')
    parser.add_argument('--exp_type', default='normal', type=str, help='normal, mix_shifts, bs1, label_shifts')
    parser.add_argument('--patch_len', default=4, type=int, help='The number of patches per row/column')
    parser.add_argument('--diversity', default=False, type=bool, help='if use the diversity loss')
    parser.add_argument('--itertimes', default=1, type=int, help='itertimes.')
    parser.add_argument('--mutual', default=True, type=bool, help='if use the mutual information.')
    parser.add_argument('--marginent', default=True, type=bool, help='if use the mutual information.')
    parser.add_argument('--exptype', default='', type=str, help='experimenttype.')
    parser.add_argument('--twomodel', default=False, type=bool, help='if use two transform images.')
    parser.add_argument('--bothcnn', default=True, type=bool, help='if two models are cnn.')

    # DeYO parameters
    parser.add_argument('--aug_type', default='patch', type=str, help='patch, pixel, occ')
    parser.add_argument('--occlusion_size', default=112, type=int)
    parser.add_argument('--row_start', default=56, type=int)
    parser.add_argument('--column_start', default=56, type=int)
    parser.add_argument('--deyo_margin', default=0.5, type=float,
                        help='Entropy threshold for sample selection $\tau_\mathrm{Ent}$ in Eqn. (8)')
    parser.add_argument('--deyo_margin_e0', default=0.4 , type=float, help='Entropy margin for sample weighting $\mathrm{Ent}_0$ in Eqn. (10)')
    parser.add_argument('--plpd_threshold', default=0.2, type=float,
                        help='PLPD threshold for sample selection $\tau_\mathrm{PLPD}$ in Eqn. (8)')
    parser.add_argument('--fishers', default=0, type=int)
    parser.add_argument('--filter_ent', default=1, type=int)
    parser.add_argument('--filter_plpd', default=1, type=int)
    parser.add_argument('--reweight_ent', default=1, type=int)
    parser.add_argument('--reweight_plpd', default=1, type=int)
    parser.add_argument('--topk', default=1000, type=int)
    parser.add_argument('--lr_mul', default=1, type=float, help='5 for Waterbirds, ColoredMNIST')
    # SAR parameters
    parser.add_argument('--sar_margin_e0', default=math.log(1000)*0.40, type=float, help='the threshold for reliable minimization in SAR, Eqn. (2)')
    parser.add_argument('--imbalance_ratio', default=500000, type=float, help='imbalance ratio for label shift exps, selected from [1, 1000, 2000, 3000, 4000, 5000, 500000], 1  denotes totally uniform and 500000 denotes (almost the same to Pure Class Order). See Section 4.3 for details;')
    parser.add_argument('--rho', default=0.05, type=float, help='rho rate')
    return parser.parse_args()


if __name__ == '__main__':
    ssl._create_default_https_context = ssl._create_unverified_context
    args = get_args()
    #loss_fct = smooth_crossentropy()
    # set random seeds
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    if not os.path.exists(args.output):  # and args.local_rank == 0
        os.makedirs(args.output, exist_ok=True)


    args.logger_name=time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())+"-{}-{}-{}-{}-level{}-{}.txt".format(args.method, args.model1,args.model2,args.itertimes, args.level, args.exptype)
    logger = get_logger(name="project", output_directory=args.output, log_name=args.logger_name, debug=False) 
        
    
    common_corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']

    if args.exp_type == 'mix_shifts':
        datasets = []
        for cpt in common_corruptions:
            args.corruption = cpt
            logger.info(args.corruption)

            val_dataset, _ = prepare_test_data(args)
            if args.method in ['tent', 'no_adapt', 'eata', 'sar','mutta']:
                val_dataset.switch_mode(True, False)
            else:
                assert False, NotImplementedError
            datasets.append(val_dataset)

        from torch.utils.data import ConcatDataset
        mixed_dataset = ConcatDataset(datasets)
        logger.info(f"length of mixed dataset us {len(mixed_dataset)}")
        val_loader = torch.utils.data.DataLoader(mixed_dataset, batch_size=args.test_batch_size, shuffle=args.if_shuffle, num_workers=args.workers, pin_memory=True)
        common_corruptions = ['mix_shifts']
 
    acc1s, acc5s = [], []
    final_acc1_net1=[]  
    final_acc1_net2=[]  
    final_acc1_orgnet=[]  
    final_acc1_combine=[]
    ir = args.imbalance_ratio
    for corrupt_i in range(0,len(common_corruptions)):
        

        torch.cuda.empty_cache()
        args.corruption = common_corruptions[corrupt_i]
        corrupt=common_corruptions[corrupt_i]
        bs = args.test_batch_size
        args.print_freq = 50000 // 20 // bs

        if args.method in ['tent', 'cotta', 'sar', 'no_adapt','coca''noadapt']:
            if args.corruption != 'mix_shifts':
                val_dataset, val_loader = prepare_test_data(args)
                val_dataset.switch_mode(True, False)
            if args.dset == 'ia':         
                args.corruption = 'ia'
                val_dataset, val_loader = prepare_test_data(args)
            if args.dset == 'ir':         
                args.corruption = 'ir'
                val_dataset, val_loader = prepare_test_data(args)
            if args.dset == 'is':         
                args.corruption = 'is'
                val_dataset, val_loader = prepare_test_data(args)
        else:
            assert False, NotImplementedError

        if args.exp_type == 'label_shifts':
            logger.info(f"imbalance ratio is {ir}")
            if args.seed == 2021:
                indices_path = './dataset/total_{}_ir_{}_class_order_shuffle_yes.npy'.format(100000, ir)
            else:
                indices_path = './dataset/seed{}_total_{}_ir_{}_class_order_shuffle_yes.npy'.format(args.seed, 100000, ir)
            logger.info(f"label_shifts_indices_path is {indices_path}")
            indices = np.load(indices_path)
            val_dataset.set_specific_subset(indices.astype(int).tolist())
        
        # build model for adaptation
        for j in range(2):
            if args.model1 == "mobilevit":
                model1 = timm.create_model("hf_hub:timm/mobilevitv2_150.cvnets_in1k", pretrained=True)
                args.lr = (0.00025 / 64) * bs * 2 if bs < 32 else 0.00025
            elif args.model1 == "beit":
                model1 = timm.create_model("hf_hub:timm/beitv2_base_patch16_224.in1k_ft_in22k_in1k", pretrained=True)
            elif args.model1 == "coatnet":
                model1 = timm.create_model("hf_hub:timm/coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k", pretrained=True)
            elif args.model1 == "Resnet18":
                model1 = Resnet.__dict__['resnet18'](pretrained=True)

                args.lr = (0.001 / 64) * bs
            elif args.model1 == "efficientvit":
                model1 =timm.create_model("hf_hub:timm/efficientnet_b0.ra4_e3600_r224_in1k", pretrained=True)
                args.lr = (0.001 / 64) * bs
            elif args.model1 == "vitbase":
                model1 = timm.create_model('vit_base_patch16_224', pretrained=True)
                model1.load_state_dict(load_file(ckpt))
                args.lr = (0.001 / 64) * bs
            elif args.model1 == "Resnet18-2":
                model1 = timm.create_model("hf_hub:timm/resnet18.fb_swsl_ig1b_ft_in1k", pretrained=True)
            elif args.model1 == "Resnet50-2":
                model1 = timm.create_model("hf_hub:timm/resnet50.fb_swsl_ig1b_ft_in1k", pretrained=True)
            elif args.model1 == "vitlarge":
                model1 = timm.create_model("hf_hub:timm/vit_large_patch14_clip_224.openai_ft_in12k_in1k", pretrained=True)

                args.lr = (0.001 / 64) * bs    
            elif args.model1 == "Resnet50":
                model1 = Resnet.__dict__['resnet50'](pretrained=True)
                args.lr = (0.00025 / 64) * bs * 2 if bs < 32 else 0.00025    
            elif args.model1 == "Resnet101":
                model1 = Resnet.__dict__['resnet101'](pretrained=True)
               
                args.lr = (0.00025 / 64) * bs * 2 if bs < 32 else 0.00025    
            else:
                assert False, NotImplementedError

            if args.model2 == "mobilevit":
                model2 = timm.create_model("hf_hub:timm/mobilevitv2_150.cvnets_in1k", pretrained=True)
                args.lr = (0.00025 / 64) * bs * 2 if bs < 32 else 0.00025
            elif args.model2 == "Resnet18":
                model2 = Resnet.__dict__['resnet18'](pretrained=True)
            elif args.model2 == "Resnet18-2":
                model2 = timm.create_model("hf_hub:timm/resnet18.fb_swsl_ig1b_ft_in1k", pretrained=True)
                args.lr = (0.001 / 64) * bs
            elif args.model2 == "efficientvit":
                model2 = timm.create_model("hf_hub:timm/efficientvit_m5.r224_in1k", pretrained=True)
                args.lr = (0.001 / 64) * bs
            elif args.model2 == "vitbase":
                model2 = timm.create_model('vit_base_patch16_224',pretrained=True)
                args.lr = (0.001 / 64) * bs
            elif args.model2 == "vitlarge":
                model2 = timm.create_model("hf_hub:timm/vit_large_patch14_clip_224.openai_ft_in12k_in1k", pretrained=True)

                args.lr = (0.001 / 64) * bs    
            elif args.model2 == "Resnet50":
                model2 = Resnet.__dict__['resnet50'](pretrained=True)
                args.lr = (0.00025 / 64) * bs * 2 if bs < 32 else 0.00025
            elif args.model2 == "Resnet50-2":
                model2 = timm.create_model("hf_hub:timm/resnet50.fb_swsl_ig1b_ft_in1k", pretrained=True)
            elif args.model2 == "Resnet101":
                model2 = Resnet.__dict__['resnet101'](pretrained=True)
                assert False, NotImplementedError
            model1 = model1.to(args.gpu)
            model2 = model2.to(args.gpu)
        

        if args.method == "noadapt":
            outputs_list_model1,targets_list=[],[]
            acc1net1top1=[]
            image=[]
            model1.eval()
            model2.eval()
            start=time.time()
            with torch.no_grad():
                for i, dl in enumerate(val_loader):
                    if args.corruption == 'ia' or args.corruption == 'ir':
                        image1,  targets = dl[0], dl[1]
                        image1 = image1.to(args.gpu2)
                        image_transform=image1
                    else:    
                        image1, image_transform, targets = dl[0], dl[1], dl[2]
                        if args.gpu is not None:
                            image1 = image1.to(args.gpu2)
                            image_transform = image_transform.to(args.gpu)
                            # images = images.to("cuda:1")
                    if torch.cuda.is_available():
                        targets = targets.to(args.gpu)
                        # target = target.to("cuda:1")
                    
                    outputsmodel1=model1(image_transform)
                    outputsmodel1 = outputsmodel1[:, imagenet_r_mask]
                    outputs_list_model1.append(outputsmodel1.detach().cpu())
                    targets_list.append(targets.cpu())
                    acc1net1, acc5net1 = accuracy(outputsmodel1, targets, topk=(1, 5))
                    acc1net1top1.append(acc1net1.item())
                    
                outputs_list_model2=torch.cat(outputs_list_model1, dim=0)
                outputs_list_model1 = torch.cat(outputs_list_model1, dim=0).numpy()
                targets_list = torch.cat(targets_list, dim=0).numpy()

            end=time.time()
            usingtime=end=start
            acc1_1=sum(acc1net1top1)/len(acc1net1top1)
            logger.info(f"Result under {args.corruption}. Original Accuracy (no adapt) is top1: {acc1_1:.5f},using time is {usingtime}")
            final_acc1_net1.append(acc1_1)
            logger.info(f"acc1s are {final_acc1_net1}")

        elif args.method =='baseline':
            del model2
            start=time.time()
            print('adapt on the ',corrupt)
            torch.cuda.reset_peak_memory_stats()  

            if args.model1 in ['Resnet18','Resnet50','Resnet101','Resnet34','Resnet50gn']:
                args.lr=0.00025*args.lrtime
            

            if args.model1 in ['mobilevit','vitbase']:
                args.lr=0.001*args.lrtime

            if args.model1 in ['vits']:
                args.lr=0.001*args.lrtime
            print(args.lr)
            if args.plusmethod=='eata':
                cor=args.corruption
                args.corruption = 'original'
                fisher_dataset, fisher_loader = prepare_test_data(args)
                fisher_dataset.set_dataset_size(args.fisher_size)
                fisher_dataset.switch_mode(True, False)
                if args.model1 in ['Resnet18','Resnet50']:
                    cnn=True
                else:
                    cnn=False
                model1fishers=calfisher(model1,args,fisher_loader,cnn=cnn)
                params, param_names = eata.collect_params(model1)
                args.corruption=cor
                optimizer = torch.optim.SGD(params, args.lr, momentum=0.9)
                adapt_model = eataorg.EATA(model1, optimizer, model1fishers, args.fisher_alpha, e_margin=args.e_margin, d_margin=args.d_margin)
                if args.dset == 'ir':
                    adapt_model.e_margin=0.4*math.log(200)

            elif args.plusmethod=='sar':
                model1=sar.configure_model(model1)
                resnet_par, resparam_names=sar.collect_params(model1)
                base_optimizer = torch.optim.SGD
                resoptimizer_sam = SAM(resnet_par, base_optimizer,rho=args.rho, lr=args.lr, momentum=0.9)
                adapt_model=sar.SAR(model1,resoptimizer_sam, margin_e0=args.sar_margin_e0)
                if args.dset == 'ir':
                    adapt_model.margin_e0=0.4*math.log(200)

            elif args.plusmethod=='none': 
                model1=tent.configure_model(model1)
                net_par, param_names=tent.collect_params(model1)
                optimizer = torch.optim.SGD(net_par, lr=args.lr,momentum=0.9)
                adapt_model=tent.Tent(model1, optimizer)
            
            elif args.plusmethod=='deyo': 
                # start=time.time()
                model1=deyo.configure_model(model1)
                net_par, param_names=deyo.collect_params(model1)
                optimizer = torch.optim.SGD(net_par, lr=args.lr,momentum=0.9)
                adapt_model=deyo.DeYO(model1,args, optimizer)

            elif args.plusmethod == "roid":
            
                model1 = roid.configure_model(model1)
                params, param_names = roid.collect_params(model1)
                logger.info(param_names)
                optimizer = torch.optim.SGD(params,args.lr, momentum=0.9)
                adapt_model = roid.ROID(args,model1, optimizer,1000)

            acc1net1top1=[]
            acc5net1top5=[]
            
            acc1orgtop1=[]
            acc5orgtop5=[]
            ece_net1_1=[]
            combineacctop1=[]
            combineacc5top5=[]
            if args.dset == 'ir':
                adapt_model.imagenet_mask = imagenet_r_mask
                
            if args.dset == 'ia':
                adapt_model.imagenet_mask = imagenet_a_mask  

                    
            for i, dl in enumerate(val_loader):
                if args.corruption == 'ia' or args.corruption == 'ir' or args.corruption == 'is':
                    image1,  targets = dl[0], dl[1]
                    image1 = image1.to(args.gpu2)
                    image_transform=image1
                else:    
                    image1, image_transform, targets = dl[0], dl[1], dl[2]
                    if args.gpu is not None:
                        image1 = image1.to(args.gpu2)
                        image_transform = image_transform.to(args.gpu)
                    # images = images.to("cuda:1")
                if torch.cuda.is_available():
                    targets = targets.to(args.gpu)
                    # target = target.to("cuda:1")
                
                if args.model1 in ['Resnet18','Resnet50','Resnet101']:
                    outputsnet1 = adapt_model(image1)
                else:
                    outputsnet1 = adapt_model(image_transform)
                acc1net1, acc5net1 = accuracy(outputsnet1, targets, topk=(1, 5))
                acc1net1top1.append(acc1net1.detach().item())
                acc5net1top5.append(acc5net1.detach().item())
                
                if i % args.print_freq == 0:
                    print('')
                    print('--------------------------------------------------')
                    print('net1 acc:',acc1net1.item(),'net1 acc5: ',acc5net1.item())
                    print(f'memory usage: {torch.cuda.max_memory_allocated()/(1024*1024):.3f}MB')
                    print('--------------------------------------------------')
                    print('')
              
            acc1_1=sum(acc1net1top1)/len(acc1net1top1)
            acc5_1=sum(acc5net1top5)/len(acc5net1top5)
            print('avg acc is ',acc1_1)
            outputs_list, targets_list = [], []
            end=time.time()
            usingtime=end-start


            print('')
            print('---------------------------finally------------------------------')
            print('net1 acc:',acc1_1,'net1 acc5: ',acc5_1)
            print('')
            logger.info(f"Result under {args.corruption}. The adaptation accuracy of mutta net1 is top1: {acc1_1} ")
            final_acc1_net1.append(acc1_1)   
    
            print('')
            print('---------------------------finally------------------------------')
            print('now is ',corrupt)
            print('net1 acc:',final_acc1_net1)
  
            print('')  
            logger.info(f"Result under {args.corruption}. The adaptation accuracy of mutta net1 is top1: {final_acc1_net1}, using{usingtime}seconds ")
        
           
        elif args.method =='coca':
            
            if args.plusmethod=='eata':
                if args.model1 in ['resnet50','resnet18','resnet101','Resnet18-2','Resnet50-2']:
                    lr1=0.00025
                else:
                    lr1=0.001
                if args.model2 in ['mobilevit','vitbase','vitlarge']:
                    lr2=0.001
                else:
                    lr2=0.00025    
                model1=eata.configure_model(model1)
                model2=eata.configure_model(model2)
                resnet_par, resparam_names=eata.collect_params(model1)
                vitnet_par, vitparam_names=eata.collect_params(model2)
                optimizer=torch.optim.SGD([
                    {'params':resnet_par,'lr':lr1,'momentum':0.9},
                    {'params':vitnet_par,'lr':lr2,'momentum':0.9}
                ])
                adapt_model=COCAV4.Coca(model1,model2, optimizer,args)
                if args.corruption == 'ir':
                    adapt_model.margin=0.4*math.log(200)            
            elif args.plusmethod=='sar':
                model1=sar.configure_model(model1)
                model2=sar.configure_model(model2)
                resnet_par, resparam_names=sar.collect_params(model1)
                vitnet_par, vitparam_names=sar.collect_params(model2)
                if args.model1 in ['resnet50','resnet18','resnet101']:
                    lr1=0.00025
                else:
                    lr1=0.001
                if args.model2 in ['mobilevit','vitbase','vitlarge']:
                    lr2=0.001
                else:
                    lr2=0.00025    
                
                base_optimizer = torch.optim.SGD
                optimizer_sam = SAM([
                    {'params':resnet_par,'lr':lr1,'momentum':0.9},
                    {'params':vitnet_par,'lr':lr2,'momentum':0.9}
                ], base_optimizer,rho=args.rho, lr=args.lr, momentum=0.9)
                
                adapt_model=COCAV4.Coca(model1,model2, optimizer_sam,args)
                if args.corruption == 'ir':
                    adapt_model.margin=0.4*math.log(200)

            elif args.plusmethod=='tent':

                model1=tent.configure_model(model1)
                model2=tent.configure_model(model2)
                resnet_par, resparam_names=tent.collect_params(model1)
                vitnet_par, vitparam_names=tent.collect_params(model2)
 
                if args.model1 in ['resnet50','resnet18','resnet101','Resnet18-2','Resnet50-2']:
                    lr1=0.00025
                else:
                    lr1=0.001
                if args.model2 in ['mobilevit','vitbase','vitlarge']:
                    lr2=0.001
                else:
                    lr2=0.00025    
                
                optimizer=torch.optim.SGD([
                    {'params':resnet_par,'lr':lr1,'momentum':0.9},
                    {'params':vitnet_par,'lr':lr2,'momentum':0.9}
                ])

                adapt_model=COCAV4.Coca(model1,model2, optimizer,args)


            start=time.time()
            acc1net1top1=[]
            acc5net1top5=[]
            
            acc1net2top1=[]
            acc5net2top5=[]
            t=[]


            acc1orgtop1=[]
            acc5orgtop5=[]
            acc_cnn1,acc_trans1,combineacc_cnn1,combineacc_trans1=[],[],[],[]
            combineacctop1=[]
            combineacc5top5=[]
            combineamount=0
            if args.corruption == 'ir':
                adapt_model.imagenet_mask = imagenet_r_mask
            if args.corruption == 'ia':
                adapt_model.imagenet_mask = imagenet_a_mask  
            torch.cuda.reset_peak_memory_stats()  
            for i, dl in enumerate(val_loader):
                if args.corruption == 'ia' or args.corruption == 'ir'or args.corruption == 'is':
                    image1,  targets = dl[0], dl[1]
                    image1 = image1.to(args.gpu2)
                    image_transform=image1
                else:    
                    image1, image_transform, targets = dl[0], dl[1], dl[2]
                    if args.gpu is not None:
                        image1 = image1.to(args.gpu2)
                        image_transform = image_transform.to(args.gpu)
                if torch.cuda.is_available():
                    targets = targets.to(args.gpu)

                
                itertimes = args.itertimes
                for j in range(itertimes):
                    if args.plusmethod=='eata':
                        outputsnet1,outputsnet2,combineoutpus = adapt_model(image1,image_transform)
                    elif args.plusmethod=='tent':
                        outputsnet1,outputsnet2,combineoutpus = adapt_model(image1,image_transform)
                    elif args.plusmethod=='sar':
                        outputsnet1,outputsnet2,combineoutpus = adapt_model(image1,image_transform)
                
            
                
                acc1net1, acc5net1 = accuracy(outputsnet1, targets, topk=(1, 5))
                acc1net2, acc5net2 = accuracy(outputsnet2, targets, topk=(1, 5))
                acc1org, acc5org = accuracy(outputsnet1, targets, topk=(1, 5))
                combineacc,combineacc5 = accuracy(combineoutpus, targets, topk=(1, 5))
                
                acc1net1=acc1net1.detach().item()
                acc1net2=acc1net2.detach().item()
                combineacc=combineacc.detach().item()
                acc1org=acc1org.detach().item()

                acc1net1top1.append(acc1net1)
                acc5net1top5.append(acc5net1)
                
                acc1net2top1.append(acc1net2)
                acc5net2top5.append(acc5net2)

                combineacctop1.append(combineacc)
                combineacc5top5.append(combineacc5)
                t.append(adapt_model.T.item())
                if i % args.print_freq == 0:
                    print('')
                    print('--------------------------------------------------')
                    print('net1 acc:',acc1net1,'net1 acc5: ',acc5net1)
                    print('net2 acc:',acc1net2,'net2 acc5: ',acc5net2)
                    print('combine acc: ',combineacc,'combine acc5: ',combineacc5)
                    print(f'memory usage: {torch.cuda.max_memory_allocated()/(1024*1024):.3f}MB')
                    print('--------------------------------------------------')
                    print('')
            
            end=time.time()        

            acc1_1=sum(acc1net1top1)/len(acc1net1top1)
            acc5_1=sum(acc5net1top5)/len(acc5net1top5)
            
            acc1_2=sum(acc1net2top1)/len(acc1net2top1)
            acc5_2=sum(acc5net2top5)/len(acc5net2top5)
            
            acc1_com=sum(combineacctop1)/len(combineacctop1)
            acc5_com=sum(combineacc5top5)/len(combineacc5top5)
            
            print('net1 avg acc is ',acc1_1)
            print('net2 avg acc is ',acc1_2)
            print('combine avg acc is ',acc1_com)
            
            outputs_list_model1,outputs_list_model2,outputs_list_combine,targets_list=[],[],[],[]

            
            print('')
            print('---------------------------finally------------------------------')
            print('net1 acc:',acc1_1,'net1 acc5: ',acc5_1)
            print('net2 acc:',acc1_2,'net2 acc5: ',acc5_2)
            print('combine acc: ',acc1_com,'combine acc5: ',acc5_com)
            print('')
            
            
            logger.info(f"Result under {args.corruption}. The adaptation accuracy of mutta net1 is top1: {acc1_1} and net2 is : {acc1_2}  combine is {acc1_com}")
            
            final_acc1_net1.append(acc1_1)   
            final_acc1_net2.append(acc1_2)   
            final_acc1_combine.append(acc1_com)

    
            print('')
            print('---------------------------finally------------------------------')
            print('now is ',corrupt)
            print('net1 acc:',final_acc1_net1)
            print('net2 acc:',final_acc1_net2)
            print('combine acc: ',final_acc1_combine)
            print('')  
            usingtime=end-start
            logger.info(f"The adaptation accuracy of mutta net1 is top1: {final_acc1_net1}  and net2 is : {final_acc1_net2}  combine is {final_acc1_combine} ,using {usingtime}seconds")
                
        elif args.method =='cotta':

            model1 = cotta.configure_model(model1)
            params, _ = cotta.collect_params(model1)
            optimizer = torch.optim.SGD(params,lr=0.005,momentum=0.9)
            adapt_model = cotta.CoTTA(model1, optimizer)
            start=time.time()
            acc1net1top1=[]
            acc5net1top5=[]
            
            acc1orgtop1=[]
            acc5orgtop5=[]
            ece_net1_1=[]
            combineacctop1=[]
            combineacc5top5=[]
            if args.dset == 'ir':
                adapt_model.imagenet_mask = imagenet_r_mask
                          
            for i, dl in enumerate(val_loader):
                if args.corruption == 'ia' or args.corruption == 'ir' or args.corruption == 'is':
                    image1,  targets = dl[0], dl[1]
                    image1 = image1.to(args.gpu2)
                    image_transform=image1
                else:    
                    image1, image_transform, targets = dl[0], dl[1], dl[2]
                    if args.gpu is not None:
                        image1 = image1.to(args.gpu2)
                        image_transform = image_transform.to(args.gpu)
                    # images = images.to("cuda:1")
                if torch.cuda.is_available():
                    targets = targets.to(args.gpu)
                    # target = target.to("cuda:1")
                torch.cuda.reset_peak_memory_stats() 

                if args.model1 in ['Resnet18','Resnet50','Resnet101']:
                    outputsnet1 = adapt_model(image1)
                else:
                    outputsnet1 = adapt_model(image_transform)

                acc1net1, acc5net1 = accuracy(outputsnet1, targets, topk=(1, 5))
                acc1net1top1.append(acc1net1.detach().item())
                acc5net1top5.append(acc5net1.detach().item())
                
                if i % args.print_freq == 0:
                    print('')
                    print('--------------------------------------------------')
                    print('net1 acc:',acc1net1.item(),'net1 acc5: ',acc5net1.item())
                    print('--------------------------------------------------')
                    print('')
                    
            acc1_1=sum(acc1net1top1)/len(acc1net1top1)
            acc5_1=sum(acc5net1top5)/len(acc5net1top5)
            outputs_list, targets_list = [], []
            # torch.save(adapt_model.net.state_dict(),'net_weight_afterTTA.pt')
            # eval_EBP(val_loader,adapt_model.net,args)
            end=time.time()
            usingtime=end-start
            print('')
            print('---------------------------finally------------------------------')
            print('net1 acc:',acc1_1,'net1 acc5: ',acc5_1)
            print('')
            logger.info(f"Result under {args.corruption}. The adaptation accuracy of mutta net1 is top1: {acc1_1} ")
            final_acc1_net1.append(acc1_1)   
            print('')
            print('---------------------------finally------------------------------')
            print('now is ',corrupt)
            print('net1 acc:',final_acc1_net1)
            
            print('')  
            logger.info(f"Result under {args.corruption}. The adaptation accuracy of mutta net1 is top1: {final_acc1_net1}, using{usingtime}seconds ")

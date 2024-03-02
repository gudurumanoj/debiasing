from enum import auto
import gc
from math import fabs
import os
import argparse
from pyexpat import model
import re
import sched
# from symbol import parameters
from copy import deepcopy       ## importing deepcopy
import torch
# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data.distributed
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
from src.helper_functions.helper_functions import mAP, CocoDetection, CutoutPIL, ModelEma, add_weight_decay
from src.models import create_model
from src.loss_functions.losses import AsymmetricLoss, BCEloss
from randaugment import RandAugment
from torch.cuda.amp import GradScaler, autocast
from torch.nn.functional import gumbel_softmax as Gumbel

parser = argparse.ArgumentParser(description='PyTorch MS_COCO Training')
parser.add_argument('--data', metavar='DIR', help='path to dataset', default='/raid/ganesh/prateekch/debiasing/MSCOCO_2014')
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--model-name', default='tresnet_m')
parser.add_argument('--model-path', default='./models_local/MS_COCO_TRresNet_M_224_81.8.pth', type=str)
parser.add_argument('--num-classes', default=80)
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')
parser.add_argument('--image-size', default=224, type=int,
                    metavar='N', help='input image size (default: 448)')
parser.add_argument('--thre', default=0.8, type=float,
                    metavar='N', help='threshold value')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('--print-freq', '-p', default=64, type=int,
                    metavar='N', help='print frequency (default: 64)')
parser.add_argument('--type','-t', default='objp', type=str)     ## added this argument to differentiate between various methods
parser.add_argument('--losscrit', type=str, default='sum')
parser.add_argument('--reg', default=0)

def main():
    args = parser.parse_args()
    args.do_bottleneck_head = False

    # Setup model
    # print('creating model...')

    if args.type == 'asl':
        print('creating model for ASL...')
        model = create_model(args).cuda()

        if args.model_path:  # make sure to load pretrained ImageNet model
            state = torch.load(args.model_path, map_location='cpu')
            filtered_dict = {k: v for k, v in state['model'].items() if
                         (k in model.state_dict() and 'head.fc' not in k)}
            model.load_state_dict(filtered_dict, strict=False)
        print('done\n')
    
    elif args.type == 'objp':
        print('creating models for obj12...')
        
        ## added two models for train and val purpose
        model_train = create_model(args).cuda()
        model_val = create_model(args).cuda()

        if args.model_path:  # make sure to load pretrained ImageNet model
            state = torch.load(args.model_path, map_location='cpu')
            filtered_dict = {k: v for k, v in state['model'].items() if
                         (k in model_train.state_dict() and 'head.fc' not in k)}
            model_train.load_state_dict(filtered_dict, strict=False)

            filtered_dict = {k: v for k, v in state['model'].items() if
                            (k in model_val.state_dict() and 'head.fc' not in k)}
            model_val.load_state_dict(filtered_dict, strict=False)

        print('done\n')

    # COCO Data loading
    instances_path_val = os.path.join(args.data, 'annotations/instances_val2014.json')
    instances_path_train = os.path.join(args.data, 'annotations/instances_train2014.json')
    # data_path_val = args.data
    # data_path_train = args.data
    data_path_val   = f'{args.data}/val2014'    # args.data
    data_path_train = f'{args.data}/train2014'  # args.data
    val_dataset = CocoDetection(data_path_val,
                                instances_path_val,
                                transforms.Compose([
                                    transforms.Resize((args.image_size, args.image_size)),
                                    transforms.ToTensor(),
                                    # normalize, # no need, toTensor does normalization
                                ]))
    train_dataset = CocoDetection(data_path_train,
                                  instances_path_train,
                                  transforms.Compose([
                                      transforms.Resize((args.image_size, args.image_size)),
                                      CutoutPIL(cutout_factor=0.5),
                                      RandAugment(),
                                      transforms.ToTensor(),
                                      # normalize,
                                  ]))
    print("len(val_dataset)): ", len(val_dataset))
    print("len(train_dataset)): ", len(train_dataset))
    
    print("Train Dataset", train_dataset.coco.getCatIds())

    # Pytorch Data loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)

    # Actuall Training
    if args.type == 'asl':
        train_multi_label_coco(model, train_loader, val_loader, args.lr)
    elif args.type == 'objp':
        ## new metalearning training loop
        # train_multi_label_coco2(model_train, model_val, train_loader, val_loader, args.lr, args)
        # learner(model_train, model_val, train_loader, val_loader, args).forward()
        learner(model_train, model_val, train_loader, val_loader, args).forwardsum()  ## look at this while training the model

def train_multi_label_coco(model, train_loader, val_loader, lr):
    ema = ModelEma(model, 0.9997)  # 0.9997^641=0.82

    # set optimizer
    Epochs = 80
    Stop_epoch = 25
    weight_decay = 1e-4
    criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True)
    parameters = add_weight_decay(model, weight_decay)
    optimizer = torch.optim.Adam(params=parameters, lr=lr, weight_decay=0)  # true wd, filter_bias_and_bn
    steps_per_epoch = len(train_loader)
    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=steps_per_epoch, epochs=Epochs,
                                        pct_start=0.2)

    highest_mAP = 0
    trainInfoList = []
    scaler = GradScaler()
    for epoch in range(Epochs):
        if epoch > Stop_epoch:
            break
        for i, (inputData, target) in enumerate(train_loader):
            inputData = inputData.cuda()
            target = target.cuda()  # (batch,3,num_classes)
            target = target.max(dim=1)[0]
            with autocast():  # mixed precision
                output = model(inputData).float()  # sigmoid will be done in loss !
            loss = criterion(output, target)
            model.zero_grad()

            scaler.scale(loss).backward()
            # loss.backward()

            scaler.step(optimizer)
            scaler.update()
            # optimizer.step()

            scheduler.step()

            ema.update(model)
            # store information
            if i % 100 == 0:
                trainInfoList.append([epoch, i, loss.item()])
                print('Epoch [{}/{}], Step [{}/{}], LR {:.1e}, Loss: {:.1f}'
                      .format(epoch, Epochs, str(i).zfill(3), str(steps_per_epoch).zfill(3),
                              scheduler.get_last_lr()[0], \
                              loss.item()))

        try:
            torch.save(model.state_dict(), os.path.join(
                'models/', 'model-{}-{}.ckpt'.format(epoch + 1, i + 1)))
        except:
            pass

        ## just normal evaluation, doesn't seem to be using any metalearning
        model.eval()
        mAP_score = validate_multi(val_loader, model, ema)
        model.train()
        if mAP_score > highest_mAP:
            highest_mAP = mAP_score
            try:
                torch.save(model.state_dict(), os.path.join(
                    'models/', 'model-highest.ckpt'))
            except:
                pass
        print('current_mAP = {:.2f}, highest_mAP = {:.2f}\n'.format(mAP_score, highest_mAP))

def validate_multi(val_loader, model, ema_model):
    print("starting validation")
    Sig = torch.nn.Sigmoid()
    preds_regular = []
    preds_ema = []
    targets = []
    for i, (input, target) in enumerate(val_loader):
        target = target
        target = target.max(dim=1)[0]
        # compute output
        with torch.no_grad():
            with autocast():
                output_regular = Sig(model(input.cuda())).cpu()
                output_ema = Sig(ema_model.module(input.cuda())).cpu()

        # for mAP calculation
        preds_regular.append(output_regular.cpu().detach())
        preds_ema.append(output_ema.cpu().detach())
        targets.append(target.cpu().detach())

    mAP_score_regular = mAP(torch.cat(targets).numpy(), torch.cat(preds_regular).numpy())
    mAP_score_ema = mAP(torch.cat(targets).numpy(), torch.cat(preds_ema).numpy())
    print("mAP score regular {:.2f}, mAP score EMA {:.2f}".format(mAP_score_regular, mAP_score_ema))
    return max(mAP_score_regular, mAP_score_ema)

## implementing without gradscaler as normal one isn't working, ot using lr scheduler as well, just plain simple training loop
## because the above one is not working 
class learner(nn.Module):
    def __init__(self, model_train, model_val, train_loader, val_loader, args):
        super(learner, self).__init__()
        self.model_train = model_train
        self.model_train_ema = ModelEma(self.model_train, 0.9997)  # 0.9997^641=0.82  # main model
        self.model_val = model_val
        self.model_val_ema = ModelEma(self.model_val, 0.9997) # 0.9997^641=0.82  # meta model @manoj : please change the nomenclature => too confusing

        self.type = args.type
        self.model_name = args.model_name

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.args = args

        self.lr = args.lr
        self.epochs = 80
        self.stop_epoch = 40    ## can be updated during training process
        self.weight_decay = 1e-4

        self.steps_per_epoch_train = len(train_loader)
        self.steps_per_epoch_val = len(val_loader)

        self.optimizer_train = torch.optim.Adam(params=self.model_train.parameters(), lr=self.lr)  # optimizer for main model
        self.optimizer_val = torch.optim.Adam(params=self.model_val.parameters(), lr=self.lr) # optimizer for meta model

        self.highest_mAP_train = 0
        self.highest_mAP_val = 0

        self.criteria = BCEloss()
        self.trainInfoList = []
        self.modeltrainvalinfo = []
        self.valInfoList = []

        self.weights = torch.rand(80).cuda() ## just initialising it randomly for the first round

        self.model_train.train()
        self.model_val.train()

    def forward(self):
        for epoch in range(self.epochs):
            print(self.weights)
            if epoch > self.stop_epoch:
                break
            
            fasttrain = deepcopy(self.model_train)
            parameters_fasttrain = add_weight_decay(fasttrain, self.weight_decay)
            optimizer_fasttrain = torch.optim.Adam(params=parameters_fasttrain, lr=self.lr, weight_decay=0)
            fasttrain.cuda()
            fasttrain.train()

            for i, (inputData, target) in enumerate(self.train_loader):
                inputData = inputData.cuda()
                target = target.cuda()
                target = target.max(dim=1)[0]
                with autocast():
                    output_train = fasttrain(inputData).float()
                loss = self.criteria(output_train, target, self.weights)
                loss = loss.sum()   # is this sum over classes [TODO: Prateek]?      ## sum over classes

                optimizer_fasttrain.zero_grad()
                loss.backward(retain_graph=True)
                optimizer_fasttrain.step()

                ## ema model update
                # self.model_train_ema.update(fasttrain)

                if i % 100 == 0:
                    # self.trainInfoList.append([epoch, i, loss.item()])
                    print('Epoch [{}/{}], Step [{}/{}], LR {:.1e}, Loss: {:.1f}'
                          .format(epoch, self.epochs, str(i).zfill(3), str(self.steps_per_epoch_train).zfill(3),
                                  self.lr, \
                                  loss.item()))
                    torch.save(fasttrain.state_dict(), os.path.join(
                    'models/{}/{}/models_train/'.format(self.type, self.model_name), 'model-{}-{}.ckpt'.format(epoch + 1, i + 1)))
            # 
            # print("-----------training loop weights before step-------------\n")
            # for name, p in self.model_train.named_parameters():
                # /print(p)
                # break
            with torch.no_grad():        
                self.optimizer_train.zero_grad()
                self.model_train.load_state_dict(fasttrain.state_dict())

            ## ema model update
            # self.model_train_ema.update(self.model_train)

            # print("-----------training loop weights after step-------------\n")
            # for name, p in self.model_train.named_parameters():
                # print(p)
                # break
            self.model_train.eval()
            for i_val, (inputData_val, target_val) in enumerate(self.val_loader):
                inputData_val = inputData_val.cuda()
                target_val = target_val.cuda()
                target_val = target_val.max(dim=1)[0]
                with autocast():
                    output_val = fasttrain(inputData_val).float()
                    # with torch.no_grad():
                    output_train_val = self.model_train(inputData_val).float()
                loss_val = self.criteria(output_val, target_val, self.weights)
                optimizer_fasttrain.zero_grad()
                loss_val = loss_val.max()
                loss_val.backward(retain_graph=True)
                optimizer_fasttrain.step()

                with torch.no_grad():
                    loss_train_val = self.criteria(output_train_val, target_val, self.weights)
                    loss_train_val = loss_train_val.sum()
                    gradients = []
                    for j, params in enumerate(fasttrain.parameters()):
                        gradients.append(deepcopy(params.grad))

                    for j, p in enumerate(self.model_val.parameters()):
                        p.grad = gradients[j]

                    self.optimizer_val.step()
                    self.optimizer_val.zero_grad()

                ## ema model update
                # self.model_val_ema.update(self.model_val)

                with autocast():
                    outputs_val = self.model_val(inputData_val).float()
                with torch.no_grad():
                    self.weights = torch.sigmoid(outputs_val.mean(dim=0))  ## updating weights

                if i_val % 100 == 0:
                    # self.valInfoList.append([epoch, i_val, loss_val.max().item()])
                    # self.modeltrainvalinfo.append([epoch, i_val, loss_train_val.item()])
                    print('Outer loop Epoch [{}/{}], Step [{}/{}], LR {:.1e}, Loss: {:.1f}'
                          .format(epoch, self.epochs, str(i_val).zfill(3), str(self.steps_per_epoch_val).zfill(3),
                                  self.lr, \
                                  loss_val.max().item()))
                    print('model_train val_loss Epoch [{}/{}], Step [{}/{}], LR {:.1e}, Loss: {:.1f}'
                            .format(epoch, self.epochs, str(i_val).zfill(3), str(self.steps_per_epoch_val).zfill(3),
                                    self.lr, \
                                    loss_train_val.item()))
                    torch.save(self.model_val.state_dict(), os.path.join(
                    'models/{}/{}/models_val/'.format(self.type, self.model_name), 'model-{}-{}.ckpt'.format(epoch + 1, i_val + 1)))
                    torch.save(self.weights, os.path.join(
                    'models/{}/{}/weights/'.format(self.type, self.model_name), 'weights-{}-{}.ckpt'.format(epoch + 1, i_val + 1)))  

            self.model_train.train()

            # print("-----------val loop gradients before-------------")
            # for i,p in enumerate(self.model_val.parameters()):
                # print(p.grad)
                # if i == 2:
                    # break

            # gradients = []
            # for i, params in enumerate(fasttrain.parameters()):
            #     gradients.append(deepcopy(params.grad))

            # # print("-----------val loop gradients after-------------",len(gradients))
            # for i, p in enumerate(self.model_val.parameters()):
            #     p.grad = gradients[i]

            # # for i,p in enumerate(self.model_val.parameters()):
            #     # print(p.grad)
            #     # if i == 2:
            #         # break
            
            # # print("-----------val loop weights before step-------------\n")
            # # for name, p in self.model_val.named_parameters():
            #     # print(p)
            #     # break
            # self.optimizer_val.step()
            # self.optimizer_val.zero_grad()

            # ## ema model update
            # self.model_val_ema.update(self.model_val)
            # print("-----------val loop weights after step-------------\n")
            # for name, p in self.model_val.named_parameters():
                # print(p)
                # break
            # with autocast():
            #     outputs_val = self.model_val(inputData_val).float()
            # self.weights = torch.sigmoid(outputs_val.mean(dim=0))  ## updating weights

            del fasttrain, parameters_fasttrain, optimizer_fasttrain
            torch.cuda.empty_cache()
            del gradients
            gc.collect()
            # torch.cuda.empty_cache()
            ## saving checkpoints
            # if i % 100 == 0:
                
            #     torch.save(self.model_val.state_dict(), os.path.join(
            #         'models/{}/{}/models_val/'.format(self.type, self.model_name), 'model-{}-{}.ckpt'.format(epoch + 1, i + 1)))
            #     torch.save(self.weights, os.path.join(
            #         'models/{}/{}/weights/'.format(self.type, self.model_name), 'weights-{}-{}.ckpt'.format(epoch + 1, i + 1)))  

            ## evaluation of the models
            print("---------evaluating the models---------\n")
            self.model_train.eval()
            self.model_val.eval()
            mAP_score_train = validate_multi(self.train_loader, self.model_train, self.model_train_ema)
            mAP_score_val = validate_multi(self.val_loader, self.model_val, self.model_val_ema)
            self.model_train.train()
            self.model_val.train()

            if mAP_score_train > self.highest_mAP_train:
                self.highest_mAP_train = mAP_score_train
                torch.save(self.model_train.state_dict(), os.path.join(
                    'models/{}/{}/models_train/'.format(self.type, self.model_name), 'model-highest.ckpt'))
            print('current_mAP = {:.2f}, highest_mAP = {:.2f}\n'.format(mAP_score_train, self.highest_mAP_train))


            if mAP_score_val > self.highest_mAP_val:
                self.highest_mAP_val = mAP_score_val
                torch.save(self.model_val.state_dict(), os.path.join(
                    'models/{}/{}/models_val/'.format(self.type, self.model_name), 'model-highest.ckpt'))    
            print('current_mAP = {:.2f}, highest_mAP = {:.2f}\n'.format(mAP_score_val, self.highest_mAP_val))

        ## printing train and val info lists
        print("-----------------train info list-----------------")
        for i in range(len(self.trainInfoList)):
            print(self.trainInfoList[i])
        print("-----------------model train val info list-----------------")
        for i in range(len(self.modeltrainvalinfo)):
            print(self.modeltrainvalinfo[i])
        print("-----------------val info list-----------------")
        for i in range(len(self.valInfoList)):
            print(self.valInfoList[i])

    ## performing the training instead of max, taking sum at all places
    def forwardsum(self):
        
        print("FORWARD SUMMING")
        for epoch in range(self.epochs):
            print(self.weights)
            #print(torch.ones(self.weights.shape))
            if epoch > self.stop_epoch:
                break
            
            fasttrain = deepcopy(self.model_train)
            parameters_fasttrain = add_weight_decay(fasttrain, self.weight_decay)
            optimizer_fasttrain = torch.optim.Adam(params=parameters_fasttrain, lr=self.lr, weight_decay=0.1)
            fasttrain.cuda()
            fasttrain.train()

            for i, (inputData, target) in enumerate(self.train_loader):
                inputData = inputData.cuda()
                target = target.cuda()
                target = target.max(dim=1)[0]
                with autocast():
                    output_train = fasttrain(inputData).float()
                loss = self.criteria(output_train, target, self.weights)
                loss = loss.sum()              # weighted sum over losses $\lambda*L_{i}$ 

                optimizer_fasttrain.zero_grad()
                loss.backward(retain_graph=True)
                optimizer_fasttrain.step()

                ## ema model update
                # self.model_train_ema.update(fasttrain)

                if i % 100 == 0:
                    # self.trainInfoList.append([epoch, i, loss.item()])
                    print('Epoch [{}/{}], Step [{}/{}], LR {:.1e}, Main Model Weighted Training Loss: {:.1f}'
                          .format(epoch, self.epochs, str(i).zfill(3), str(self.steps_per_epoch_train).zfill(3),
                                  self.lr, \
                                  loss.item()))
                    # torch.save(fasttrain.state_dict(), os.path.join(
                    # 'models/{}/{}/models_train/'.format(self.type, self.model_name), 'model-{}-{}.ckpt'.format(epoch + 1, i + 1)))
            # 
            # print("-----------training loop weights before step-------------\n")
            # for name, p in self.model_train.named_parameters():
                # /print(p)
                # break
            with torch.no_grad():        
                self.optimizer_train.zero_grad()
                self.model_train.load_state_dict(fasttrain.state_dict())

            ## ema model update
            # self.model_train_ema.update(self.model_train)

            # print("-----------training loop weights after step-------------\n")
            # for name, p in self.model_train.named_parameters():
                # print(p)
                # break
                
            # TODO: Prateek - Check validation data stats
            
            
            # fasttrain is mainly for validation set checking of the main model.
            
            self.model_train.eval()
            for i_val, (inputData_val, target_val) in enumerate(self.val_loader):
                inputData_val = inputData_val.cuda()
                target_val = target_val.cuda()
                target_val = target_val.max(dim=1)[0]
                with autocast():
                    output_val = fasttrain(inputData_val).float()  # validation set output using the main model  
                    # with torch.no_grad():
                    # output_train_val = self.model_train(inputData_val).float()
                
                
                loss_val = self.criteria(output_val, target_val, self.weights) # validation loss
                loss_val_unweighted = self.criteria(output_val,target_val, torch.ones(self.weights.shape).cuda())
                optimizer_fasttrain.zero_grad()
                
                loss_val_unweighted = loss_val_unweighted.sum()
                
                if self.args.losscrit == 'sum': # the difference is just this statement
                    #print("Validation Loss length", len(loss_val))
                    loss_val = loss_val.sum()
                else:
                    loss_val = Gumbel(loss_val, tau=0.01, hard=False, eps=1e-10, dim=-1)
                
                loss_val.backward(retain_graph=True)
                optimizer_fasttrain.step()
                
                # Main Model on Validation Set
                
                with torch.no_grad():
                    with autocast():
                        output_train_val = self.model_train(inputData_val).float()  # main model output on validation set
                    loss_train_val = self.criteria(output_train_val, target_val, self.weights) # main model loss on validation set
                    loss_train_val_unweighted = self.criteria(output_train_val,target_val, torch.ones(self.weights.shape).cuda())
                    loss_train_val_unweighted = loss_train_val_unweighted.sum()
                    loss_train_val = loss_train_val.sum()
                    gradients = []
                    for j, params in enumerate(fasttrain.parameters()):
                        gradients.append(deepcopy(params.grad))

                    for j, p in enumerate(self.model_val.parameters()):
                        p.grad = gradients[j]

                    self.optimizer_val.step()
                    self.optimizer_val.zero_grad()
                
                with autocast():
                    outputs_val = self.model_val(inputData_val).float() # main model output on the val set.
                with torch.no_grad():
                    self.weights = torch.sigmoid(outputs_val.mean(dim=0))
                
                if i_val % 100 == 0:
                    # self.valInfoList.append([epoch, i_val, loss_val.max().item()])
                    # self.modeltrainvalinfo.append([epoch, i_val, loss_train_val.item()])
                    print('Outer loop Epoch Maximum [{}/{}], Step [{}/{}], LR {:.1e}, Meta Learning Summed up Validation Loss: {:.1f}'
                          .format(epoch, self.epochs, str(i_val).zfill(3), str(self.steps_per_epoch_val).zfill(3),
                                  self.lr, \
                                  loss_val.sum().item()))
                    print('model_train val_loss Epoch [{}/{}], Step [{}/{}], LR {:.1e}, Main Model Validation Loss: {:.1f}'
                            .format(epoch, self.epochs, str(i_val).zfill(3), str(self.steps_per_epoch_val).zfill(3),
                                    self.lr, \
                                    loss_train_val.item()))
                    
                    print('model_train val_loss Epoch [{}/{}], Step [{}/{}], LR {:.1e}, Main Model Unweighted Validation Loss: {:.1f}'
                            .format(epoch, self.epochs, str(i_val).zfill(3), str(self.steps_per_epoch_val).zfill(3),
                                    self.lr, \
                                    loss_train_val_unweighted.item()))
                    # torch.save(self.model_val.state_dict(), os.path.join(
                    # 'models/{}/{}/models_val/'.format(self.type, self.model_name), 'model-{}-{}.ckpt'.format(epoch + 1, i_val + 1)))
                    # torch.save(self.weights, os.path.join(
                    # 'models/{}/{}/weights/'.format(self.type, self.model_name), 'weights-{}-{}.ckpt'.format(epoch + 1, i_val + 1)))
                
            # self.model_train.train()
            del fasttrain, parameters_fasttrain, optimizer_fasttrain
            torch.cuda.empty_cache()
            del gradients
            gc.collect()

            print("---------evaluating the models---------\n")
            self.model_train.eval()
            self.model_val.eval()
            mAP_score_train = validate_multi(self.train_loader, self.model_train, self.model_train_ema)
            mAP_score_val = validate_multi(self.val_loader, self.model_val, self.model_val_ema)
            self.model_train.train()
            self.model_val.train()
            if mAP_score_train > self.highest_mAP_train:
                self.highest_mAP_train = mAP_score_train
                # torch.save(self.model_train.state_dict(), os.path.join(
                #     'models/{}/{}/models_train/'.format(self.type, self.model_name), 'model-highest.ckpt'))
            print('current_mAP = {:.2f}, highest_mAP = {:.2f}\n'.format(mAP_score_train, self.highest_mAP_train))

            if mAP_score_val > self.highest_mAP_val:
                self.highest_mAP_val = mAP_score_val
                # torch.save(self.model_val.state_dict(), os.path.join(
                    # 'models/{}/{}/models_val/'.format(self.type, self.model_name), 'model-highest.ckpt'))
            print('current_mAP = {:.2f}, highest_mAP = {:.2f}\n'.format(mAP_score_val, self.highest_mAP_val))
    
if __name__ == '__main__':
    main()
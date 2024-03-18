from enum import auto
import gc
from math import fabs
import os
import argparse
from pyexpat import model
import numpy as np
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
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--print-info', default='no', type=str)
parser.add_argument('--meta-lr', default=1e-4, type=float)



def main():
    args = parser.parse_args()
    ## random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
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
    
    elif 'objp' in args.type or args.type == 'bce':  ## should later change this args.type = 'wtbce' correctly 
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
    
    # print("Train Dataset", train_dataset.coco.getCatIds())

    ## to get dataset stats (id to class mapping, frequency of classes)
    # print("------------------- Train Dataset -------------------")
    # print("Train Dataset", train_dataset.cat2cat)
    

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
    elif args.type == 'objpmax':
        learner(model_train, model_val, train_loader, val_loader, args).forward()
    elif args.type == 'objpsum':
        ## new metalearning training loop
        # train_multi_label_coco2(model_train, model_val, train_loader, val_loader, args.lr, args)
        # learner(model_train, model_val, train_loader, val_loader, args).forward()
        learner(model_train, model_val, train_loader, val_loader, args).forwardsum()  ## look at this while training the model
    elif args.type == 'objpsum2':
        learner(model_train, model_val, train_loader, val_loader, args).forwardsum2()
    elif args.type == 'bce':
        bcelearner(model_train, train_loader, val_loader, args).forward()
    elif args.type == 'objpsum_ct':
        metalearner(model_train, model_val, train_loader, val_loader, args).forward()

def train_multi_label_coco(model, train_loader, val_loader, lr):
    ema = ModelEma(model, 0.9997)  # 0.9997^641=0.82

    # set optimizer
    Epochs = 80
    Stop_epoch = 40
    weight_decay = 1e-4
    criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True)
    parameters = add_weight_decay(model, weight_decay)
    optimizer = torch.optim.Adam(params=parameters, lr=lr, weight_decay=0)  # true wd, filter_bias_and_bn
    steps_per_epoch = len(train_loader)
    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=steps_per_epoch, valEpocws=Epochs,
                                        pct_start=0.2)

    highest_mAP = 0
    trainInfoList = []
    scaler = GradScaler()
    for epoch in range(Epochs):
        if epoch > Stop_epoch:
            break
        # print("lr ", scheduler.get_last_lr())
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

        # try:
        #     torch.save(model.state_dict(), os.path.join(
        #         'models/', 'model-{}-{}.ckpt'.format(epoch + 1, i + 1)))
        # except:
        #     pass

        ## just normal evaluation, doesn't seem to be using any metalearning
        model.eval()

        for i_val, (inputData_val, target_val) in enumerate(val_loader):
            inputData_val = inputData_val.cuda()
            target_val = target_val.cuda()
            target_val = target_val.max(dim=1)[0]
            with autocast():
                output_val = model(inputData_val).float()
            loss_val = criterion(output_val, target_val)
            if i_val % 100 == 0:
                print('Val Loss valEpocw [{}/{}], Step [{}/{}], LR {:.1e}, Loss: {:.1f}'
                      .format(epoch, Epochs, str(i_val).zfill(3), str(len(val_loader)).zfill(3),
                              scheduler.get_last_lr()[0], \
                              loss_val.item()))

        mAP_score = validate_multi(val_loader, model, ema)
        model.train()
        if mAP_score > highest_mAP:
            highest_mAP = mAP_score
            # try:
            #     torch.save(model.state_dict(), os.path.join(
            #         'models/', 'model-highest.ckpt'))
            # except:
            #     pass
        print('current_mAP = {:.2f}, highest_mAP = {:.2f}'.format(mAP_score, highest_mAP))

def validate_multi(val_loader, model, ema_model, print_info="yes", name = ''):
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

    mAP_score_regular = mAP(torch.cat(targets).numpy(), torch.cat(preds_regular).numpy(),print_info)
    mAP_score_ema = mAP(torch.cat(targets).numpy(), torch.cat(preds_ema).numpy(),print_info)
    print("mAP score regular {:.2f}, mAP score EMA {:.2f}".format(mAP_score_regular, mAP_score_ema))
    return max(mAP_score_regular, mAP_score_ema)


def printClassLoss(loss, name):
    print("--------------------------------")
    for i in range(80):
        print("{} Class_wise {} Loss: {}".format(name, i, loss[i]))

## bcelearner, similar to asl learner, but with bce loss
class bcelearner(nn.Module):
    def __init__(self, model_train, train_loader, val_loader, args):
        super(bcelearner, self).__init__()
        self.model_train = model_train
        self.model_train_ema = ModelEma(self.model_train, 0.9997)  # 0.9997^641=0.82  # main model

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.args = args

        self.criteria = BCEloss()
        # self.traininfoList = []
        # self.valinfoList = []

        self.type = args.type
        self.model_name = args.model_name

        ## optimizer and scheduler
        self.epoch = 80
        self.stop_epoch = 40
        self.weight_decay = 1e-4
        self.parameters = add_weight_decay(self.model_train, self.weight_decay)
        self.optimizer = torch.optim.Adam(params=self.parameters, lr=args.lr, weight_decay=0)
        self.scheduler = lr_scheduler.OneCycleLR(self.optimizer, max_lr=args.lr, steps_per_epoch=len(train_loader), epochs=self.epoch, pct_start=0.2)
        self.scaler = GradScaler()

        self.highest_mAP_train = 0
        self.highest_mAP_val = 0

        self.model_train.train()

    def forward(self):
        for epoch in range(self.epoch):
            if epoch>self.stop_epoch:
                break
            
            print("lr: ", self.scheduler.get_last_lr()) 
            self.model_train.train()

            for i, (inputData, target) in enumerate(self.train_loader):
                inputData = inputData.cuda()
                
                target = target.cuda()
                target = target.max(dim=1)[0]
                # print("Target", target.shape)
                with autocast():
                    output_train = self.model_train(inputData).float()
                loss = self.criteria(output_train, target, torch.ones(80).cuda())

                if i%100 == 0 and self.args.print_info == "yes":
                    # printClassLoss(loss, "BCE_Train")
                    print("BCE Train Loss: ", loss)

                loss = loss.sum()

                self.model_train.zero_grad()

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                self.scheduler.step()

                self.model_train_ema.update(self.model_train)

                if i % 100 == 0:
                    # self.traininfoList.append([epoch, i, loss.item()])
                    print('Epoch [{}/{}], Step [{}/{}], LR {:.1e}, Loss: {:.1f}'
                          .format(epoch, self.epoch, str(i).zfill(3), str(len(self.train_loader)).zfill(3),
                                  self.scheduler.get_last_lr()[0], \
                                  loss.item()))
            
            ## evaluation of the models
            self.model_train.eval()

            for i_val, (inputData_val, target_val) in enumerate(self.val_loader):
                inputData_val = inputData_val.cuda()
                target_val = target_val.cuda()
                target_val = target_val.max(dim=1)[0]
                with autocast():
                    output_val = self.model_train(inputData_val).float()
                loss_val = self.criteria(output_val, target_val, torch.ones(80).cuda())

                if i_val % 100 == 0 and self.args.print_info == "yes":
                    # printClassLoss(loss_val, "BCE_Val")
                    print("BCE Val Loss: ", loss_val)

                loss_val = loss_val.sum()

                if i_val % 100 == 0:
                    # self.valinfoList.append([epoch, i_val, loss_val.item().cpu()])
                    print('Val loss valEpocw [{}/{}], Step [{}/{}], LR {:.1e}, Loss: {:.1f}'
                          .format(epoch, self.epoch, str(i_val).zfill(3), str(len(self.val_loader)).zfill(3),
                                  self.scheduler.get_last_lr()[0], \
                                  loss_val.item()))

            mAP_score_train = validate_multi(self.train_loader, self.model_train, self.model_train_ema, self.args.print_info)
            mAP_score_val = validate_multi(self.val_loader, self.model_train, self.model_train_ema, self.args.print_info)

            if mAP_score_train > self.highest_mAP_train:
                self.highest_mAP_train = mAP_score_train
            if mAP_score_val > self.highest_mAP_val:
                self.highest_mAP_val = mAP_score_val

            print('Train_data_mAP: current_mAP = {:.2f}, highest_mAP = {:.2f}'.format(mAP_score_train, self.highest_mAP_train))
            print('Val_data_mAP: current_mAP = {:.2f}, highest_mAP = {:.2f}'.format(mAP_score_val, self.highest_mAP_val))

        


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
        self.meta_lr = args.meta_lr
        self.epochs = 80
        self.stop_epoch = 40    ## can be updated during training process
        self.weight_decay = 1e-4

        self.steps_per_epoch_train = len(train_loader)
        self.steps_per_epoch_val = len(val_loader)

        self.optimizer_train = torch.optim.Adam(params=self.model_train.parameters(), lr=self.lr)  # optimizer for main model
        self.optimizer_val = torch.optim.Adam(params=self.model_val.parameters(), lr=self.meta_lr) # optimizer for meta model

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
        print('FORWARD MAX-ING')
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

                if i % 100 == 0 and self.args.print_info == "yes":
                    # printClassLoss(loss, "Max_Train")
                    print("Max Train Loss: ", loss)
                # print(loss.shape)
                loss = loss.max()   # this is max      ## max over classes

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
                    # torch.save(fasttrain.state_dict(), os.path.join(
                    # 'models/{}/{}/models_train/'.format(self.type, self.model_name), 'model-{}-{}.ckpt'.format(epoch + 1, i + 1)))
           
            with torch.no_grad():        
                self.optimizer_train.zero_grad()
                self.model_train.load_state_dict(fasttrain.state_dict())

            # ema model update
            self.model_train_ema.update(self.model_train)

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

                if i_val % 100 == 0 and self.args.print_info == "yes":
                    # printClassLoss(loss_val, "Max_Val Meta Model")
                    print("Max_Val Meta Model: ", loss_val)

                loss_val = loss_val.max()
                loss_val.backward(retain_graph=True)
                optimizer_fasttrain.step()

                with torch.no_grad():
                    loss_train_val = self.criteria(output_train_val, target_val, self.weights)

                    if i_val % 100 == 0 and self.args.print_info == "yes":
                        # printClassLoss(loss_train_val, "Max_Val Main Model Weighted Val Loss")
                        print("Max_Val Main Model Weighted Val Loss: ", loss_train_val)

                    loss_train_val = loss_train_val.max()
                    loss_train_val_unweighted = self.criteria(output_train_val,target_val, torch.ones(self.weights.shape).cuda())
                    
                    if i_val % 100 == 0 and self.args.print_info == "yes":
                        # printClassLoss(loss_train_val_unweighted, "Max_Val Main Model Unweighted Val Loss")
                        print("Max_Val Main Model Unweighted Val Loss: ", loss_train_val_unweighted)

                    loss_train_val_unweighted = loss_train_val_unweighted.max()

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
                    print('Outer loop valEpocw Maximum [{}/{}], Step [{}/{}], LR {:.1e}, Meta Learning Max Validation Loss: {:.1f}'
                            .format(epoch, self.epochs, str(i_val).zfill(3), str(self.steps_per_epoch_val).zfill(3),
                                    self.optimizer_val.param_groups[0]['lr'], \
                                    loss_val.item()))
                    print('model_train val_loss valEpocw [{}/{}], Step [{}/{}], LR {:.1e}, Main Model Weighted Validation Loss: {:.1f}'
                            .format(epoch, self.epochs, str(i_val).zfill(3), str(self.steps_per_epoch_val).zfill(3),
                                    self.lr, \
                                    loss_train_val.item()))
                    print('model_train val_loss  valEpocw [{}/{}], Step [{}/{}], LR {:.1e}, Main Model Unweighted Validation Loss: {:.1f}'
                            .format(epoch, self.epochs, str(i_val).zfill(3), str(self.steps_per_epoch_val).zfill(3),
                                    self.lr, \
                                    loss_train_val_unweighted.item()))
                    # torch.save(self.model_val.state_dict(), os.path.join(
                    # 'models/{}/{}/models_val/'.format(self.type, self.model_name), 'model-{}-{}.ckpt'.format(epoch + 1, i_val + 1)))
                    # torch.save(self.weights, os.path.join(
                    # 'models/{}/{}/weights/'.format(self.type, self.model_name), 'weights-{}-{}.ckpt'.format(epoch + 1, i_val + 1)))  


            del fasttrain, parameters_fasttrain, optimizer_fasttrain
            torch.cuda.empty_cache()
            del gradients
            gc.collect()
            
            ## evaluation of the models
            print("---------evaluating the models---------\n")
            self.model_train.eval()
            self.model_val.eval()
            mAP_score_train = validate_multi(self.train_loader, self.model_train, self.model_train_ema, self.args.print_info)
            mAP_score_val = validate_multi(self.val_loader, self.model_train, self.model_train_ema, self.args.print_info)
            self.model_train.train()
            self.model_val.train()

            if mAP_score_train > self.highest_mAP_train:
                self.highest_mAP_train = mAP_score_train
                # torch.save(self.model_train.state_dict(), os.path.join(
                #     'models/{}/{}/models_train/'.format(self.type, self.model_name), 'model-highest.ckpt'))
            print('Train_data_mAP: current_mAP = {:.2f}, highest_mAP = {:.2f}'.format(mAP_score_train, self.highest_mAP_train))


            if mAP_score_val > self.highest_mAP_val:
                self.highest_mAP_val = mAP_score_val
                # torch.save(self.model_val.state_dict(), os.path.join(
                #     'models/{}/{}/models_val/'.format(self.type, self.model_name), 'model-highest.ckpt'))    
            print('Val_data_mAP: current_mAP = {:.2f}, highest_mAP = {:.2f}'.format(mAP_score_val, self.highest_mAP_val))


    ## performing the training instead of max, taking sum at all places
    def forwardsum(self):
        
        print("FORWARD SUMMING")
        for epoch in range(self.epochs):
            print(self.weights)
            #print(torch.ones(self.weights.shape))
            if epoch > self.stop_epoch:
                break
            
            # meta model
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

                if i % 100 == 0 and self.args.print_info == "yes":
                    # printClassLoss(loss, "Sum_Train")
                    print("Sum Train Loss: ", loss)

                loss = loss.sum()              # weighted sum over losses $\lambda*L_{i}$ 

                optimizer_fasttrain.zero_grad()
                loss.backward(retain_graph=True)
                optimizer_fasttrain.step()

                if i % 100 == 0:
                    # self.trainInfoList.append([epoch, i, loss.item()])
                    print('Epoch [{}/{}], Step [{}/{}], LR {:.1e}, Main Model Weighted Training Loss: {:.1f}'
                          .format(epoch, self.epochs, str(i).zfill(3), str(self.steps_per_epoch_train).zfill(3),
                                  self.lr, \
                                  loss.item()))
                    # torch.save(fasttrain.state_dict(), os.path.join(
                    # 'models/{}/{}/models_train/'.format(self.type, self.model_name), 'model-{}-{}.ckpt'.format(epoch + 1, i + 1)))
            
            with torch.no_grad():        
                self.optimizer_train.zero_grad()
                self.model_train.load_state_dict(fasttrain.state_dict())

            # ema model update
            self.model_train_ema.update(self.model_train)
                
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
                # loss_val_unweighted = self.criteria(output_val,target_val, torch.ones(self.weights.shape).cuda())
                optimizer_fasttrain.zero_grad()
                
                # loss_val_unweighted = loss_val_unweighted.sum()
                
                if self.args.losscrit == 'sum': # the difference is just this statement
                    #print("Validation Loss length", len(loss_val))
                    if i_val % 100 == 0 and self.args.print_info == "yes":
                        # printClassLoss(loss_val, "Sum_Val Meta Model")
                        print("Sum_Val Meta Model: ", loss_val)
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
                    
                    if i_val % 100 == 0 and self.args.print_info == "yes":
                        # printClassLoss(loss_train_val, "Sum_Val Main Model Weighted Val Loss")
                        print("Sum_Val Main Model Weighted Val Loss: ", loss_train_val)

                    loss_train_val = loss_train_val.sum()

                    if i_val % 100 == 0 and self.args.print_info == "yes":
                        # printClassLoss(loss_train_val_unweighted, "Sum_Val Main Model Unweighted Val Loss")
                        print("Sum_Val Main Model Unweighted Val Loss: ", loss_train_val_unweighted)
                    loss_train_val_unweighted = loss_train_val_unweighted.sum()
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
                    print('Outer loop valEpocw Maximum [{}/{}], Step [{}/{}], LR {:.1e}, Meta Learning Summed up Validation Loss: {:.1f}'
                          .format(epoch, self.epochs, str(i_val).zfill(3), str(self.steps_per_epoch_val).zfill(3),
                                  self.optimizer_val.param_groups[0]['lr'], \
                                  loss_val.sum().item()))
                    print('model_train val_loss valEpocw [{}/{}], Step [{}/{}], LR {:.1e}, Main Model Validation Loss: {:.1f}'
                            .format(epoch, self.epochs, str(i_val).zfill(3), str(self.steps_per_epoch_val).zfill(3),
                                    self.lr, \
                                    loss_train_val.item()))
                    
                    print('model_train val_loss valEpocw [{}/{}], Step [{}/{}], LR {:.1e}, Main Model Unweighted Validation Loss: {:.1f}'
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
            mAP_score_train = validate_multi(self.train_loader, self.model_train, self.model_train_ema, self.args.print_info)
            mAP_score_val = validate_multi(self.val_loader, self.model_train, self.model_train_ema, self.args.print_info)
            self.model_train.train()
            self.model_val.train()
            if mAP_score_train > self.highest_mAP_train:
                self.highest_mAP_train = mAP_score_train
                # torch.save(self.model_train.state_dict(), os.path.join(
                #     'models/{}/{}/models_train/'.format(self.type, self.model_name), 'model-highest.ckpt'))
            print('Train_data_mAP: current_mAP = {:.2f}, highest_mAP = {:.2f}'.format(mAP_score_train, self.highest_mAP_train))

            if mAP_score_val > self.highest_mAP_val:
                self.highest_mAP_val = mAP_score_val
                # torch.save(self.model_val.state_dict(), os.path.join(
                    # 'models/{}/{}/models_val/'.format(self.type, self.model_name), 'model-highest.ckpt'))
            print('Val_data_mAP: current_mAP = {:.2f}, highest_mAP = {:.2f}'.format(mAP_score_val, self.highest_mAP_val))

## similar to learner class but with different implementation (hopefully it'll be the correct implementation)
## implementation of this isn't complete yet
class metalearner(nn.Module):
    def __init__(self, model_train, model_val, train_loader, val_loader, args):
        super(metalearner, self).__init__()

        ## models
        self.model_train = model_train
        self.model_train_ema = ModelEma(self.model_train, 0.9997)
        self.model_val = model_val
        self.model_val_ema = ModelEma(self.model_val, 0.9997)

        ## dataloaders
        self.train_loader = train_loader
        self.val_loader = val_loader

        ## args
        self.args = args

        ## training params
        self.lr = args.lr
        self.epochs = 80
        self.stop_epoch = 40
        self.weight_decay = 1e-4

        ## steps per epoch
        self.steps_per_epoch_train = len(train_loader)
        self.steps_per_epoch_val = len(val_loader)

        ## criteria, optimizer and scheduler for model_train and model_val
        self.criteria = BCEloss()

        self.parameters_train = add_weight_decay(self.model_train, self.weight_decay)
        self.optimizer_train = torch.optim.Adam(params=self.parameters_train, lr=self.lr, weight_decay=0)
        self.scheduler_train = lr_scheduler.OneCycleLR(self.optimizer_train, max_lr=self.lr, steps_per_epoch=self.steps_per_epoch_train, epochs=self.epochs, pct_start=0.2)

        self.parameters_val = add_weight_decay(self.model_val, self.weight_decay)
        self.optimizer_val = torch.optim.Adam(params=self.parameters_val, lr=self.lr, weight_decay=0)
        self.scheduler_val = lr_scheduler.OneCycleLR(self.optimizer_val, max_lr=self.lr, steps_per_epoch=self.steps_per_epoch_val, epochs=self.epochs, pct_start=0.2)

        ## scaler
        self.scaler_train = GradScaler()
        self.scaler_val = GradScaler()

        ## highest mAP scores
        self.highest_mAP_train = 0
        self.highest_mAP_val = 0

        ## lists to store training and validation info
        self.trainInfoList = []
        self.valInfoList = []

        ## weights
        self.weights = torch.rand(80).cuda()

    def forward(self):
        print("FORWARD Summing updated weights")

        

        for epoch in range(self.epochs):
            if epoch > self.stop_epoch:
                break
            
            print(self.weights)
            ## training model_train
            for i, (inputData, target) in enumerate(self.train_loader):
                inputData = inputData.cuda()
                target = target.cuda()
                target = target.max(dim=1)[0]
                with autocast():
                    output_train = self.model_train(inputData).float()
                loss = self.criteria(output_train, target, self.weights)
                loss = loss.sum()

                self.model_train.zero_grad()
                self.scaler_train.scale(loss).backward(retain_graph = True)
                self.scaler_train.step(self.optimizer_train)
                self.scaler_train.update()
                self.scheduler_train.step()

                ## model_train ema update
                self.model_train_ema.update(self.model_train)

                if i % 100 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], LR {:.1e}, Main Model Weighted Training Loss: {:.1f}'
                          .format(epoch, self.epochs, str(i).zfill(3), str(self.steps_per_epoch_train).zfill(3),
                                  self.scheduler_train.get_last_lr()[0], \
                                  loss.item()))

            ## evaluating model_train
            self.model_train.eval()
            for i_val, (inputData_val, target_val) in enumerate(self.val_loader):
                inputData_val = inputData_val.cuda()
                target_val = target_val.cuda()
                target_val = target_val.max(dim=1)[0]
                with autocast():
                    output_val = self.model_train(inputData_val).float()
                    weight = self.model_val(inputData_val).float()
                weight_ = torch.sigmoid(weight.mean(dim=0))
                loss_val = self.criteria(output_val, target_val, weight_)
                loss_val = loss_val.sum()

                self.model_val.zero_grad()
                self.scaler_val.scale(loss_val).backward(retain_graph = True)
                self.scaler_val.step(self.optimizer_val)
                self.scaler_val.update()
                self.scheduler_val.step()

                ## val model ema update
                self.model_val_ema.update(self.model_val)

                if i_val % 100 == 0:
                    print('Val loss valEpocw [{}/{}], Step [{}/{}], LR {:.1e}, Main Model Weighted Validation Loss: {:.1f}'
                          .format(epoch, self.epochs, str(i_val).zfill(3), str(self.steps_per_epoch_val).zfill(3),
                                  self.scheduler_val.get_last_lr()[0], \
                                  loss_val.item()))

                    loss_val_val = weight.sum()
                    print('Val loss valEpocw [{}/{}], Step [{}/{}], LR {:.1e}, Meta Model weighted Validation Loss: {:.1f}'
                          .format(epoch, self.epochs, str(i_val).zfill(3), str(self.steps_per_epoch_val).zfill(3),
                                  self.scheduler_val.get_last_lr()[0], \
                                  loss_val_val.item()))
                    
                    loss_train_unweighted = self.criteria(output_val, target_val, torch.ones(self.weights.shape).cuda())
                    loss_train_unweighted = loss_train_unweighted.sum()
                    print('Val loss valEpocw [{}/{}], Step [{}/{}], LR {:.1e}, Main Model Unweighted Validation Loss: {:.1f}'
                          .format(epoch, self.epochs, str(i_val).zfill(3), str(self.steps_per_epoch_val).zfill(3),
                                  self.scheduler_val.get_last_lr()[0], \
                                  loss_train_unweighted.item()))

            ## updating weights
            self.weights = weight_

            ## evaluating the models
            self.model_train.eval()
            mAP_score_train = validate_multi(self.train_loader, self.model_train, self.model_train_ema, self.args.print_info)
            mAP_score_val = validate_multi(self.val_loader, self.model_train, self.model_train_ema, self.args.print_info)
            
            self.model_train.train()
            self.model_val.train()

            if mAP_score_train > self.highest_mAP_train:
                self.highest_mAP_train = mAP_score_train
            print('Train_data_mAP: current_mAP = {:.2f}, highest_mAP = {:.2f}'.format(mAP_score_train, self.highest_mAP_train))

            if mAP_score_val > self.highest_mAP_val:
                self.highest_mAP_val = mAP_score_val
            print('Val_data_mAP: current_mAP = {:.2f}, highest_mAP = {:.2f}'.format(mAP_score_val, self.highest_mAP_val))
    
if __name__ == '__main__':
    main()
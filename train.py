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

from cocostats import coco2014


parser = argparse.ArgumentParser(description='PyTorch MS_COCO Training')
parser.add_argument('--data', metavar='DIR', help='path to dataset', default='/raid/ganesh/prateekch/MSCOCO_2014')
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
parser.add_argument('--print-info', default='yes', type=str)
parser.add_argument('--meta-lr', default=1e-4, type=float)

def getInverseClassFreqs():
    rootPath = '/raid/ganesh/prateekch/MSCOCO_2014'
    train_annotation_file_path = os.path.join(rootPath, 'annotations/instances_train2014.json')
    val_annotation_file_path = os.path.join(rootPath, 'annotations/instances_val2014.json')

    coco2014_train = coco2014(train_annotation_file_path)
    coco2014_val = coco2014(val_annotation_file_path)

    trainClassFreqs = coco2014_train.class_frequencies
    invClassFreqs = torch.tensor(10000/np.array(list(trainClassFreqs.values())), requires_grad=False, device='cuda')
    return invClassFreqs

def getlearntweights():

    w = torch.tensor([8.5983e-03, 3.3228e-04, 1.6858e-03, 3.2429e-04, 3.4240e-04, 4.3736e-04,
        1.5994e-04, 1.0868e-04, 5.4978e-04, 1.8753e-04, 1.8216e-04, 4.5452e-04,
        1.2122e-04, 7.1922e-04, 9.4465e-04, 4.1021e-04, 5.8142e-04, 5.3437e-03,
        1.0822e-02, 4.6157e-04, 7.1747e-05, 4.3498e-04, 6.3778e-04, 3.9378e-04,
        1.0643e-03, 9.7466e-05, 1.8190e-03, 1.3778e-04, 6.1190e-05, 1.3345e-03,
        6.5805e-04, 1.2390e-04, 1.2549e-02, 3.1616e-03, 5.0841e-03, 3.0605e-01,
        6.4207e-05, 6.2014e-04, 1.4639e-02, 8.4071e-04, 8.2097e-02, 4.6225e-01,
        9.9109e-01, 1.9012e-01, 9.6806e-01, 9.4148e-01, 2.1047e-04, 4.2092e-04,
        2.2199e-03, 4.0909e-04, 1.1029e-03, 1.8684e-03, 1.4722e-03, 7.6037e-04,
        3.4434e-04, 1.4308e-03, 6.7108e-02, 6.9280e-03, 4.5576e-02, 8.4404e-04,
        8.6304e-01, 1.6499e-04, 3.8882e-02, 4.4598e-03, 2.2276e-02, 1.5684e-02,
        2.4597e-02, 2.2216e-02, 8.0488e-01, 3.6187e-03, 1.2733e-04, 2.2566e-04,
        6.0695e-01, 2.0614e-02, 9.9623e-01, 1.0000e+00, 1.0000e+00, 1.0000e+00,
        9.9995e-01, 3.9083e-03], device='cuda:0')
    
    return w

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
    
    elif 'objp' in args.type or 'bce' in args.type:  ## should later change this args.type = 'wtbce' correctly 
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

    elif args.type == 'objw':  ## using the model parameters from the bce trained model and for learning lambdas, the normal imagenet pretrained model
        print('creating models for obj with starting model parameters as bce trained ...')

        ## added two models for train and val purpose
        model_train = create_model(args).cuda()
        model_val = create_model(args).cuda()

        if args.model_path:  # make sure to load pretrained ImageNet model
            state = torch.load('/raid/ganesh/prateekch/debiasing/models_local/model_highest_bce40epochs.pth', map_location='cpu')
            filtered_dict = {k: v for k, v in state.items() if
                         (k in model_train.state_dict() and 'head.fc' not in k)}
            model_train.load_state_dict(filtered_dict, strict=False)

            state2 = torch.load(args.model_path, map_location='cpu')
            filtered_dict = {k: v for k, v in state2['model'].items() if
                            (k in model_val.state_dict() and 'head.fc' not in k)}
            model_val.load_state_dict(filtered_dict, strict=False)
        
        print('done\n')

    elif args.type == 'objw2':
        print('creating models for obj with starting model parameters as bce trained ...')

        ## added two models for train and val purpose
        model_train = create_model(args).cuda()
        model_val = create_model(args).cuda()

        if args.model_path:
            state = torch.load('/raid/ganesh/prateekch/debiasing/models_local/model_highest_bce40epochs.pth', map_location='cpu')
            filtered_dict = {k: v for k, v in state.items() if
                         (k in model_train.state_dict() and 'head.fc' not in k)}
            model_train.load_state_dict(filtered_dict, strict=False)

            state2 = torch.load('/raid/ganesh/prateekch/debiasing/models_local/model_highest_bce40epochs.pth', map_location='cpu')
            filtered_dict = {k: v for k, v in state2.items() if
                            (k in model_val.state_dict() and 'head.fc' not in k)}
            model_val.load_state_dict(filtered_dict, strict=False)

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
    elif args.type == 'objpsum' or args.type == 'objpinv' or args.type == 'objpinv2' or 'objw' in args.type :
        learner(model_train, model_val, train_loader, val_loader, args).forwardsum() 
    elif args.type == 'bce':
        bcelearner(model_train, train_loader, val_loader, args).forward()
    elif args.type == 'bce2':
        bcelearner(model_train, train_loader, val_loader, args).forward()

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
    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=steps_per_epoch, epochs=Epochs,
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

        try:
            torch.save(model.state_dict(), os.path.join(
                'models/', 'model-{}-{}.ckpt'.format(epoch + 1, i + 1)))
        except:
            pass

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
            try:
                torch.save(model.state_dict(), os.path.join(
                    'models/', 'model-highest.ckpt'))
            except:
                pass
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

        if args.type == 'bce2':
            self.weights = getlearntweights().cuda()
        else:
            self.weights = torch.ones(80).cuda()

    def forward(self):
        for epoch in range(self.epoch):
            if epoch>self.stop_epoch: ## change this to stop_epoch later
                break
            
            # print("lr: ", self.scheduler.get_last_lr()) 
            self.model_train.train()

            for i, (inputData, target) in enumerate(self.train_loader):
                inputData = inputData.cuda()
                
                target = target.cuda()
                target = target.max(dim=1)[0]
                # print("Target", target.shape)
                with autocast():
                    output_train = self.model_train(inputData).float()
                loss = self.criteria(output_train, target, self.weights)

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
                loss_val = self.criteria(output_val, target_val, self.weights)

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
                ## saving the model
                torch.save(self.model_train.state_dict(), os.path.join('models_local_new', 'model_highest_bce40epochs.pth'))

            print('Train_data_mAP: current_mAP = {:.2f}, highest_mAP = {:.2f}'.format(mAP_score_train, self.highest_mAP_train))
            print('Val_data_mAP: current_mAP = {:.2f}, highest_mAP = {:.2f}'.format(mAP_score_val, self.highest_mAP_val))

class learner(nn.Module):
    def __init__(self, model_train, model_val, train_loader, val_loader, args):
        super(learner, self).__init__()
        self.model_train = model_train
        self.model_train_ema = ModelEma(self.model_train, 0.9997)  # 0.9997^641=0.82  # main model
        self.model_val = model_val
        self.model_val_ema = ModelEma(self.model_val, 0.9997) # 0.9997^641=0.82  # meta model 

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

        if args.type == 'objpinv':
            self.weights = getInverseClassFreqs()
        else:
            self.weights = torch.rand(80).cuda() ## just initialising it randomly for the first round

        self.invClassFreqs = getInverseClassFreqs().cuda()

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
                    print("Max Train Loss: ", loss)
                loss = loss.max()   # this is max      ## max over classes

                optimizer_fasttrain.zero_grad()
                loss.backward(retain_graph=True)
                optimizer_fasttrain.step()


                if i % 100 == 0:
                    # self.trainInfoList.append([epoch, i, loss.item()])
                    print('Epoch [{}/{}], Step [{}/{}], LR {:.1e}, Loss: {:.1f}'
                          .format(epoch, self.epochs, str(i).zfill(3), str(self.steps_per_epoch_train).zfill(3),
                                  self.lr, \
                                  loss.item()))
                    torch.save(fasttrain.state_dict(), os.path.join(
                     'models/{}/{}/models_train/'.format(self.type, self.model_name), 'model-{}-{}.ckpt'.format(epoch + 1, i + 1)))
           
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
                    output_train_val = self.model_train(inputData_val).float()

                
                loss_val = self.criteria(output_val, target_val, self.weights)
                optimizer_fasttrain.zero_grad()

                if i_val % 100 == 0 and self.args.print_info == "yes":
                    print("Max_Val Meta Model: ", loss_val)

                loss_val = loss_val.max()
                loss_val.backward(retain_graph=True)
                optimizer_fasttrain.step()

                with torch.no_grad():
                    loss_train_val = self.criteria(output_train_val, target_val, self.weights)

                    if i_val % 100 == 0 and self.args.print_info == "yes":
                        print("Max_Val Main Model Weighted Val Loss: ", loss_train_val)

                    loss_train_val = loss_train_val.max()
                    loss_train_val_unweighted = self.criteria(output_train_val,target_val, torch.ones(self.weights.shape).cuda())
                    
                    if i_val % 100 == 0 and self.args.print_info == "yes":
                        print("Max_Val Main Model Unweighted Val Loss: ", loss_train_val_unweighted)

                    loss_train_val_unweighted = loss_train_val_unweighted.max()

                    gradients = []
                    for j, params in enumerate(fasttrain.parameters()):
                        gradients.append(deepcopy(params.grad))

                    for j, p in enumerate(self.model_val.parameters()):
                        p.grad = gradients[j]

                    self.optimizer_val.step()
                    self.optimizer_val.zero_grad()


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
                    torch.save(self.model_val.state_dict(), os.path.join(
                     'models/{}/{}/models_val/'.format(self.type, self.model_name), 'model-{}-{}.ckpt'.format(epoch + 1, i_val + 1)))
                    torch.save(self.weights, os.path.join(
                        'models/{}/{}/weights/'.format(self.type, self.model_name), 'weights-{}-{}.ckpt'.format(epoch + 1, i_val + 1)))  


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
                torch.save(self.model_train.state_dict(), os.path.join(
                     'models/{}/{}/models_train/'.format(self.type, self.model_name), 'model-highest.ckpt'))
            print('Train_data_mAP: current_mAP = {:.2f}, highest_mAP = {:.2f}'.format(mAP_score_train, self.highest_mAP_train))


            if mAP_score_val > self.highest_mAP_val:
                self.highest_mAP_val = mAP_score_val
                torch.save(self.model_val.state_dict(), os.path.join(
                     'models/{}/{}/models_val/'.format(self.type, self.model_name), 'model-highest.ckpt'))    
            print('Val_data_mAP: current_mAP = {:.2f}, highest_mAP = {:.2f}'.format(mAP_score_val, self.highest_mAP_val))


    ## performing the training instead of max, taking sum at all places
    def forwardsum(self):
        
        print("FORWARD SUMMING")
        for epoch in range(self.epochs):
            print(self.weights)
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
                    torch.save(fasttrain.state_dict(), os.path.join(
                     'models/{}/{}/models_train/'.format(self.type, self.model_name), 'model-{}-{}.ckpt'.format(epoch + 1, i + 1)))
            
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
                    output_val = fasttrain(inputData_val).float()  # validation set output using the main model  
                
                if self.args.type == 'objpinv':
                    loss_val = self.criteria(output_val, target_val, torch.ones(self.weights.shape).cuda())
                    loss_val = loss_val*self.invClassFreqs
                else:
                    loss_val = self.criteria(output_val, target_val, self.weights)
                optimizer_fasttrain.zero_grad()
                
                
                if self.args.losscrit == 'sum': 
                    if i_val % 100 == 0 and self.args.print_info == "yes":
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
                        print("Sum_Val Main Model Weighted Val Loss: ", loss_train_val)

                    loss_train_val = loss_train_val.sum()

                    if i_val % 100 == 0 and self.args.print_info == "yes":
                        print("Sum_Val Main Model Unweighted Val Loss: ", loss_train_val_unweighted)
                    loss_train_val_unweighted = loss_train_val_unweighted.sum()
                    gradients = []
                    for j, params in enumerate(fasttrain.parameters()):
                        gradients.append(deepcopy(params.grad))

                    for j, p in enumerate(self.model_val.parameters()):
                        p.grad = gradients[j]

                    self.optimizer_val.step()
                    self.optimizer_val.zero_grad()
                

                if self.args.type != 'objpinv':
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
                    torch.save(self.model_val.state_dict(), os.path.join(
                     'models/{}/{}/models_val/'.format(self.type, self.model_name), 'model-{}-{}.ckpt'.format(epoch + 1, i_val + 1)))
                    torch.save(self.weights, os.path.join(
                     'models/{}/{}/weights/'.format(self.type, self.model_name), 'weights-{}-{}.ckpt'.format(epoch + 1, i_val + 1)))
                
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
                torch.save(self.model_train.state_dict(), os.path.join(
                     'models/{}/{}/models_train/'.format(self.type, self.model_name), 'model-highest.ckpt'))
            print('Train_data_mAP: current_mAP = {:.2f}, highest_mAP = {:.2f}'.format(mAP_score_train, self.highest_mAP_train))

            if mAP_score_val > self.highest_mAP_val:
                self.highest_mAP_val = mAP_score_val
                torch.save(self.model_val.state_dict(), os.path.join(
                    'models/{}/{}/models_val/'.format(self.type, self.model_name), 'model-highest.ckpt'))
            print('Val_data_mAP: current_mAP = {:.2f}, highest_mAP = {:.2f}'.format(mAP_score_val, self.highest_mAP_val))


if __name__ == '__main__':
    main()
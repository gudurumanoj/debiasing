from enum import auto
import gc
from math import fabs
import os
import argparse
from pyexpat import model
import re
import sched
from symbol import parameters
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



parser = argparse.ArgumentParser(description='PyTorch MS_COCO Training')
parser.add_argument('data', metavar='DIR', help='path to dataset', default='/home/MSCOCO_2014/')
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--model-name', default='tresnet_m')
parser.add_argument('--model-path', default='./tresnet_m.pth', type=str)
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

parser.add_argument('--type','-t', default='asl', type=str)     ## added this argument to differentiate between various methods


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
        learner(model_train, model_val, train_loader, val_loader, args).forward()


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


## implementing without gradscaler as normal one isn't working, ot using lr scheduler as well, just plain simple training loop
## because the above one is not working 
class learner(nn.Module):
    def __init__(self, model_train, model_val, train_loader, val_loader, args):
        super(learner, self).__init__()
        self.model_train = model_train
        self.model_train_ema = ModelEma(self.model_train, 0.9997)  # 0.9997^641=0.82
        self.model_val = model_val
        self.model_val_ema = ModelEma(self.model_val, 0.9997) # 0.9997^641=0.82

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

        self.optimizer_train = torch.optim.Adam(params=self.model_train.parameters(), lr=self.lr)
        self.optimizer_val = torch.optim.Adam(params=self.model_val.parameters(), lr=self.lr)

        self.highest_mAP_train = 0
        self.highest_mAP_val = 0

        self.criteria = BCEloss()
        self.trainInfoList = []
        self.valInfoList = []

        self.weights = torch.rand(80).cuda() ## just initialising it randomly for the first round

        self.model_train.train()
        self.model_val.train()


    def forward(self):
        for epoch in range(self.epochs):
            # print(self.weights)
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
                loss = loss.sum()                                   ## sum over classes

                optimizer_fasttrain.zero_grad()
                loss.backward(retain_graph=True)
                optimizer_fasttrain.step()

                if i % 100 == 0:
                    self.trainInfoList.append([epoch, i, loss.item()])
                    print('Epoch [{}/{}], Step [{}/{}], LR {:.1e}, Loss: {:.1f}'
                          .format(epoch, self.epochs, str(i).zfill(3), str(self.steps_per_epoch_train).zfill(3),
                                  self.lr, \
                                  loss.item()))
            # 
            # print("-----------training loop weights before step-------------\n")
            # for name, p in self.model_train.named_parameters():
                # /print(p)
                # break
                    
            self.optimizer_train.zero_grad()
            self.model_train.load_state_dict(fasttrain.state_dict())

            ## ema model update
            self.model_train_ema.update(self.model_train)

            # print("-----------training loop weights after step-------------\n")
            # for name, p in self.model_train.named_parameters():
                # print(p)
                # break

            for i_val, (inputData_val, target_val) in enumerate(self.val_loader):
                inputData_val = inputData_val.cuda()
                target_val = target_val.cuda()
                target_val = target_val.max(dim=1)[0]
                with autocast():
                    output_val = fasttrain(inputData_val).float()
                loss_val = self.criteria(output_val, target_val, self.weights)
                optimizer_fasttrain.zero_grad()
                loss_val = loss_val.max()
                loss_val.backward(retain_graph=True)
                optimizer_fasttrain.step()

                if i_val % 100 == 0:
                    self.valInfoList.append([epoch, i_val, loss_val.max().item()])
                    print('Outer loop Epoch [{}/{}], Step [{}/{}], LR {:.1e}, Loss: {:.1f}'
                          .format(epoch, self.epochs, str(i_val).zfill(3), str(self.steps_per_epoch_val).zfill(3),
                                  self.lr, \
                                  loss_val.max().item()))

            # print("-----------val loop gradients before-------------")
            # for i,p in enumerate(self.model_val.parameters()):
                # print(p.grad)
                # if i == 2:
                    # break

            gradients = []
            for i, params in enumerate(fasttrain.parameters()):
                gradients.append(deepcopy(params.grad))

            # print("-----------val loop gradients after-------------",len(gradients))
            for i, p in enumerate(self.model_val.parameters()):
                p.grad = gradients[i]

            # for i,p in enumerate(self.model_val.parameters()):
                # print(p.grad)
                # if i == 2:
                    # break
            
            # print("-----------val loop weights before step-------------\n")
            # for name, p in self.model_val.named_parameters():
                # print(p)
                # break
            self.optimizer_val.step()
            self.optimizer_val.zero_grad()

            ## ema model update
            self.model_val_ema.update(self.model_val)
            # print("-----------val loop weights after step-------------\n")
            # for name, p in self.model_val.named_parameters():
                # print(p)
                # break
            with autocast():
                outputs_val = self.model_val(inputData_val).float()
            self.weights = torch.sigmoid(outputs_val.mean(dim=0))  ## updating weights

            del fasttrain, parameters_fasttrain, optimizer_fasttrain
            torch.cuda.empty_cache()
            del gradients
            gc.collect()
            # torch.cuda.empty_cache()
            ## saving checkpoints
            if i % 100 == 0:
                torch.save(self.model_train.state_dict(), os.path.join(
                    'models/{}/{}/models_train/'.format(self.type, self.model_name), 'model-{}-{}.ckpt'.format(epoch + 1, i + 1)))
                torch.save(self.model_val.state_dict(), os.path.join(
                    'models/{}/{}/models_val/'.format(self.type, self.model_name), 'model-{}-{}.ckpt'.format(epoch + 1, i + 1)))
                torch.save(self.weights, os.path.join(
                    'models/{}/{}/weights/'.format(self.type, self.model_name), 'weights-{}-{}.ckpt'.format(epoch + 1, i + 1)))  

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
        for i in range(len(self.trainInfoList)):
            print(self.trainInfoList[i])
        for i in range(len(self.valInfoList)):
            print(self.valInfoList[i])

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


if __name__ == '__main__':
    main()

# def train_multi_label_coco2(model_train, model_val, train_loader, val_loader, lr, args):

#     """
#     model train: model for training
#     model val: model for learning weights (aka \lambdas)
#     Consists of meta learning loop
#     """

#     ## added this to smoothen the training and validation models weights
#     ema_train = ModelEma(model_train, 0.9997)  # 0.9997^641=0.82
#     ema_val = ModelEma(model_val, 0.9997)  # 0.9997^641=0.82

#     # set optimizer, similar attributes except for criterion
#     Epochs = 80
#     Stop_epoch = 40   ## should set it later by finetuning it
#     weight_decay = 1e-4
#     # criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True)
    
#     ## training model parameters
#     criterion_train = BCEloss()
#     parameters_train = add_weight_decay(model_train, weight_decay)
#     optimizer_train = torch.optim.Adam(params=parameters_train, lr=lr, weight_decay=0)  # true wd, filter_bias_and_bn
#     scheduler_train = lr_scheduler.OneCycleLR(optimizer_train, max_lr=lr, steps_per_epoch=len(train_loader), epochs=Epochs,
#                                         pct_start=0.2)
    
#     ## validation (metalearning) model parameters
#     parameters_val = add_weight_decay(model_val, weight_decay)
#     optimizer_val = torch.optim.Adam(params=parameters_val, lr=lr, weight_decay=0)  # true wd, filter_bias_and_bn
#     scheduler_val = lr_scheduler.OneCycleLR(optimizer_val, max_lr=lr, steps_per_epoch=len(val_loader), epochs=Epochs,
#                                         pct_start=0.2)
#     ## no need of criterion for validation model, as it is just learning weights for the training model
    
#     steps_per_epoch_train = len(train_loader)
#     steps_per_epoch_val = len(val_loader)

#     highest_mAP_train = 0
#     highest_mAP_val = 0

#     trainInfoList = []
#     valInfoList = []

#     scaler_train = GradScaler()
#     scaler_val = GradScaler()

#     weights = torch.rand(80).cuda() ## just initialising it randomly for the first round

#     for epoch in range(Epochs):
#         if epoch > Stop_epoch:
#             break
        
#         for i, (inputData, target) in enumerate(train_loader):
            
#             ## training model_train
#             inputData = inputData.cuda()
#             target = target.cuda()  # (batch,3,num_classes)
#             target = target.max(dim=1)[0]
#             with autocast():  # mixed precision
#                 outputs_train = model_train(inputData).float()  # sigmoid will be done in loss !
#             loss = criterion_train(outputs_train, target, weights)   
#             loss = loss.sum()                                   ## sum over classes
#             # model_train.zero_grad()

#             # scaler_train.scale(loss).backward(retain_graph=True)
#             scaler_train.scale(loss).backward(retain_graph=True)     ## will update it if it doesn't work

#             # loss.backward()

#             scaler_train.step(optimizer_train)
#             scaler_train.update()
#             # optimizer.step()

#             scheduler_train.step()
#             model_train.zero_grad()
#             ema_train.update(model_train) ## no ema here, but can be added for both train and val models
#             # model_train.zero_grad()
#             # store information
#             if i % 100 == 0:
#                 trainInfoList.append([epoch, i, loss.item()])
#                 print('Epoch [{}/{}], Step [{}/{}], LR {:.1e}, Loss: {:.1f}'
#                       .format(epoch, Epochs, str(i).zfill(3), str(steps_per_epoch_train).zfill(3),
#                               scheduler_train.get_last_lr()[0], \
#                               loss.item()))

#         ## validation model_val
#         # fasttrain = deepcopy(model_train.to(torch.device('cpu')))   ## this will be used to update the weights of model_val 
#         #                                     ## because we don't want to update the weights of model_train when we do backward()
#         #                                     ## during the validation (or metalearning) loop
#         # fasttrain = type(model_train)()  # Create a new instance of the same model class
#         # fasttrain = create_model(args).cuda()
#         fasttrain = create_model(args).cuda()
                
#         try:
#             fasttrain.load_state_dict(model_train.state_dict())  # Copy the weights from model_train
#         except:
#             print("error in loading the weights")
#         # statedict = model_train.state_dict()

#         # filtereddict = {k: v for k, v in statedict.items() if 'head.fc' not in k}
#         # fasttrain.load_state_dict(filtereddict)  # Copy the weights from model_train
#         # fasttrain.load_state_dict(model_train.state_dict()).cuda()  # Copy the weights from model_train
# # fasttrain.to(torch.device('cpu'))  # Move the model to the CPU
#         scaler_fasttrain = GradScaler()
#         parameters_fasttrain = add_weight_decay(fasttrain, weight_decay)
#         optimizer_fasttrain = torch.optim.Adam(params=parameters_fasttrain, lr=lr, weight_decay=0)  # true wd, filter_bias_and_bn
#         scheduler_fasttrain = lr_scheduler.OneCycleLR(optimizer_fasttrain, max_lr=lr, steps_per_epoch=len(val_loader), epochs=Epochs,
#                                         pct_start=0.2)
        
#         # fasttrain.cuda()


#         for i_val, (inputData_val, target_val) in enumerate(val_loader):
#             gradients = []
#             inputData_val = inputData_val.cuda()
#             target_val = target_val.cuda()
#             target_val = target_val.max(dim=1)[0]
#             with autocast():
#                 # outputs_val = model_val(inputData_val).float() ## should be done after updating the val model
#                 outputs_train = fasttrain(inputData_val).float()
            
#             # Calculate the validation loss
#             loss_val = criterion_train(outputs_train, target_val, weights)

#             ## backward pass and calculates the gradients which are used to update the weights of model_val
#             scaler_fasttrain.scale(loss_val.max()).backward(retain_graph=True)
#             scaler_fasttrain.step(optimizer_fasttrain)
#             scaler_fasttrain.update()
#             scheduler_fasttrain.step()

#             model_val.zero_grad()

#             # ## updating the weights of model_val
#             for i, params in enumerate(fasttrain.parameters()):
#             #     if i == 0:
#                 gradients.append(deepcopy(params.grad.to(torch.device('cpu'))))
#             #     else:
#             # gradients.append(deepcopy(fasttrain.parameters.grad.to(torch.device('cpu'))))
#             for i, p in enumerate(model_val.parameters()):
#                 p.grad = gradients[i].cuda()
            
#             ## updating the weights of model_val
#             # for p_global, p_local in zip(model_val.parameters(), fasttrain.parameters()):
#             #         p_global.grad += p_local.grad
            
#             # scaler_val.step(optimizer_val)
#             optimizer_val.step()
#             # scaler_val.update()
#             # model_val.zero_grad()  
#             scheduler_val.step()

#             # del gradients
#             # del fasttrain
#             # gc.collect()
            
#             # Backward pass lossmax and updating the parameters of model_val
#             # # loss_max.requires_grad = True  ## dont know if should be doing this or not
#             # model_val.zero_grad()
#             # # scaler.scale(torch.max(loss_val)).backward()
#             # # scaler_val.scale(loss_val.max()).backward(retain_graph=True)

#             # # scaler_val.step(optimizer_val)
#             # # scaler_val.update()
#             # scheduler_val.step()

#             ## should look at weight updates once
#             # # store information
#             if i_val % 100 == 0: 
#                 valInfoList.append([epoch, i_val, loss_val.max().item()])
#                 print('Outer loop Epoch [{}/{}], Step [{}/{}], LR {:.1e}, Loss: {:.1f}'
#                       .format(epoch, Epochs, str(i_val).zfill(3), str(steps_per_epoch_val).zfill(3),
#                               scheduler_val.get_last_lr()[0], \
#                               loss_val.max().item()))
        
#         with autocast():
#             outputs_val = model_val(inputData_val).float()
#         weights = torch.sigmoid(outputs_val.mean(dim=0))  ## updating weights
#         print("weights: ", weights)

            


#         try:
#             torch.save(model_train.state_dict(), os.path.join(
#                 'models/{}/{}/models_train/'.format(args.type, args.model_name), 'model-{}-{}.ckpt'.format(epoch + 1, i + 1)))  ## changed path here, remember this
#         except:
#             pass

#         try:
#             torch.save(model_val.state_dict(), os.path.join(
#                 'models/{}/{}/models_val/'.format(args.type ,args.model_name), 'model-{}-{}.ckpt'.format(epoch + 1, i + 1)))  ## changed path here, remember this
#         except:
#             pass


        
#         ## should write a validation loop here, donot forget to do it
#         model_train.eval()
#         mAP_score_train = validate_multi(val_loader, model_train, ema_train)
#         model_train.train()
#         if mAP_score_train > highest_mAP_train:
#             highest_mAP_train = mAP_score_train
#             try:
#                 torch.save(model_train.state_dict(), os.path.join(
#                     'models/{}/{}/models_train'.format(args.type ,args.model_name), 'model-highest.ckpt'))    
#             except:
#                 pass
#         print('current_mAP = {:.2f}, highest_mAP = {:.2f}\n'.format(mAP_score_train, highest_mAP_train))

#         model_val.eval()
#         mAP_score_val = validate_multi(val_loader, model_val, ema_train)
#         model_val.train()
#         if mAP_score_val > highest_mAP_val:
#             highest_mAP_val = mAP_score_val
#             try:
#                 torch.save(model_val.state_dict(), os.path.join(
#                     'models/{}/{}/models_val'.format(args.type ,args.model_name), 'model-highest.ckpt'))    
#             except:
#                 pass
#         print('current_mAP = {:.2f}, highest_mAP = {:.2f}\n'.format(mAP_score_val, highest_mAP_val))

#     ## if printed correctly then directly write it to a file
#     for i in range(len(trainInfoList)):
#         print(trainInfoList[i])

#     for i in range(len(valInfoList)):
#         print(valInfoList[i])

#     # ## writing to a csv file, may be put this into a function? 
#     ##(a helper function for writing to csv file which takes in the list and the path to write to)
#     # with open(os.path.join('models/{}/{}/trainInfo.csv'.format(type,model_name)), 'w') as f:
#     #     for i in range(len(trainInfoList)):
#     #         f.write(trainInfoList[i])
#     # with open(os.path.join('models/{}/{}/valInfo.csv'.format(type,model_name)), 'w') as f:
#     #     for i in range(len(valInfoList)):
#     #         f.write(valInfoList[i])

        

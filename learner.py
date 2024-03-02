import gc
import os
from copy import deepcopy       ## importing deepcopy
import torch
# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data.distributed
from src.helper_functions.helper_functions import ModelEma, add_weight_decay
from src.loss_functions.losses import AsymmetricLoss, BCEloss
from torch.cuda.amp import autocast
from torch.nn.functional import gumbel_softmax as Gumbel

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
                loss = loss.sum()                                   ## sum over classes

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
        for epoch in range(self.epochs):
            print(self.weights)
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
                loss = loss.sum()                                   ## sum over classes

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
                    # output_train_val = self.model_train(inputData_val).float()
                loss_val = self.criteria(output_val, target_val, self.weights)
                optimizer_fasttrain.zero_grad()
                loss_val = Gumbel(loss_val, tau=0.01, hard=False, eps=1e-10, dim=-1) ## the difference is just this statement
                loss_val.backward(retain_graph=True)
                optimizer_fasttrain.step()
                with torch.no_grad():
                    with autocast():
                        output_train_val = self.model_train(inputData_val).float()
                    loss_train_val = self.criteria(output_train_val, target_val, self.weights)
                    loss_train_val = loss_train_val.sum()
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
                    self.weights = torch.sigmoid(outputs_val.mean(dim=0))
                
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
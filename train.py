from enum import auto
import os
import argparse
import sched
from symbol import parameters
import torch
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
        train_multi_label_coco2(model_train, model_val, train_loader, val_loader, args.lr, args.model_name, args.type)


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


def train_multi_label_coco2(model_train, model_val, train_loader, val_loader, lr, model_name, type):

    ema_train = ModelEma(model_train, 0.9997)  # 0.9997^641=0.82


    # set optimizer, similar attributes except for criterion
    Epochs = 80
    Stop_epoch = 40
    weight_decay = 1e-4
    # criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True)
    
    ## training model parameters
    criterion_train = BCEloss()
    parameters_train = add_weight_decay(model_train, weight_decay)
    optimizer_train = torch.optim.Adam(params=parameters_train, lr=lr, weight_decay=0)  # true wd, filter_bias_and_bn
    scheduler_train = lr_scheduler.OneCycleLR(optimizer_train, max_lr=lr, steps_per_epoch=len(train_loader), epochs=Epochs,
                                        pct_start=0.2)
    
    ## validation (metalearning) model parameters
    parameters_val = add_weight_decay(model_val, weight_decay)
    optimizer_val = torch.optim.Adam(params=parameters_val, lr=lr, weight_decay=0)  # true wd, filter_bias_and_bn
    scheduler_val = lr_scheduler.OneCycleLR(optimizer_val, max_lr=lr, steps_per_epoch=len(val_loader), epochs=Epochs,
                                        pct_start=0.2)
    criterion_val = BCEloss()
    
    steps_per_epoch_train = len(train_loader)
    steps_per_epoch_val = len(val_loader)

    highest_mAP = 0

    trainInfoList = []
    valInfoList = []

    scaler_train = GradScaler()
    scaler_val = GradScaler()

    weights = torch.rand(80).cuda() ## just initialising it randomly for the first round

    for epoch in range(Epochs):
        if epoch > Stop_epoch:
            break
        for i, (inputData, target) in enumerate(train_loader):
            
            ## training model_train
            inputData = inputData.cuda()
            target = target.cuda()  # (batch,3,num_classes)
            target = target.max(dim=1)[0]
            with autocast():  # mixed precision
                outputs_train = model_train(inputData).float()  # sigmoid will be done in loss !
            loss = criterion_train(outputs_train, target, weights)   
            loss = loss.sum()                                   ## sum over classes
            model_train.zero_grad()

            scaler_train.scale(loss).backward(retain_graph=True)
            # loss.backward()

            scaler_train.step(optimizer_train)
            scaler_train.update()
            # optimizer.step()

            scheduler_train.step()
            ema_train.update(model_train) ## no ema here, but can be added for both train and val models

            # store information
            if i % 100 == 0:
                trainInfoList.append([epoch, i, loss.item()])
                print('Epoch [{}/{}], Step [{}/{}], LR {:.1e}, Loss: {:.1f}'
                      .format(epoch, Epochs, str(i).zfill(3), str(steps_per_epoch_train).zfill(3),
                              scheduler_train.get_last_lr()[0], \
                              loss.item()))

        ## validation model_val
        for i_val, (inputData_val, target_val) in enumerate(val_loader):
            inputData_val = inputData_val.cuda()
            target_val = target_val.cuda()
            target_val = target_val.max(dim=1)[0]
            with autocast():
                outputs_val = model_val(inputData_val).float()
                outputs_train = model_train(inputData_val).float()
            
            # Calculate the validation loss
            loss_val = criterion_train(outputs_train, target_val, weights)
            
            # Backward pass lossmax and updating the parameters of model_val
            # loss_max.requires_grad = True  ## dont know if should be doing this or not
            model_val.zero_grad()
            # scaler.scale(torch.max(loss_val)).backward()
            scaler_val.scale(loss_val.max()).backward(retain_graph=True)

            # scaler_val.step(optimizer_val)
            # scaler_val.update()
            scheduler_val.step()

            ## should look at weight updates once
            weights = torch.sigmoid(outputs_val.mean(dim=0))  ## updating weights

            # store information
            if i % 100 == 0:
                valInfoList.append([epoch, i_val, loss_val.max().item()])
                print('Epoch [{}/{}], Step [{}/{}], LR {:.1e}, Loss: {:.1f}'
                      .format(epoch, Epochs, str(i_val).zfill(3), str(steps_per_epoch_val).zfill(3),
                              scheduler_val.get_last_lr()[0], \
                              loss_val.max().item()))


        try:
            torch.save(model_train.state_dict(), os.path.join(
                'models/{}/{}/models_train/'.format(type, model_name), 'model-{}-{}.ckpt'.format(epoch + 1, i + 1)))  ## changed path here, remember this
        except:
            pass

        try:
            torch.save(model_val.state_dict(), os.path.join(
                'models/{}/{}/models_val/'.format(type ,model_name), 'model-{}-{}.ckpt'.format(epoch + 1, i + 1)))  ## changed path here, remember this
        except:
            pass


        
        ## should write a validation loop here, donot forget to do it
        model_train.eval()
        mAP_score = validate_multi(val_loader, model_train, ema_train)
        model_train.train()
        if mAP_score > highest_mAP:
            highest_mAP = mAP_score
            try:
                torch.save(model_train.state_dict(), os.path.join(
                    'models/{}/{}/models_train'.format(type ,model_name), 'model-highest.ckpt'))    
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


if __name__ == '__main__':
    main()

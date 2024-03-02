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

        

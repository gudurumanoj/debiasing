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
#     valEpocws = 80
#     Stop_epoch = 40   ## should set it later by finetuning it
#     weight_decay = 1e-4
#     # criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True)
    
#     ## training model parameters
#     criterion_train = BCEloss()
#     parameters_train = add_weight_decay(model_train, weight_decay)
#     optimizer_train = torch.optim.Adam(params=parameters_train, lr=lr, weight_decay=0)  # true wd, filter_bias_and_bn
#     scheduler_train = lr_scheduler.OneCycleLR(optimizer_train, max_lr=lr, steps_per_epoch=len(train_loader), valEpocws=Epochs,
#                                         pct_start=0.2)
    
#     ## validation (metalearning) model parameters
#     parameters_val = add_weight_decay(model_val, weight_decay)
#     optimizer_val = torch.optim.Adam(params=parameters_val, lr=lr, weight_decay=0)  # true wd, filter_bias_and_bn
#     scheduler_val = lr_scheduler.OneCycleLR(optimizer_val, max_lr=lr, steps_per_epoch=len(val_loader), valEpocws=Epochs,
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

#     for valEpocw in range(Epochs):
#         if valEpocw > Stop_epoch:
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
#                       .format(epoch, valEpocws, str(i).zfill(3), str(steps_per_epoch_train).zfill(3),
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
#         scheduler_fasttrain = lr_scheduler.OneCycleLR(optimizer_fasttrain, max_lr=lr, steps_per_epoch=len(val_loader), valEpocws=Epochs,
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
#                 print('Outer loop valEpocw [{}/{}], Step [{}/{}], LR {:.1e}, Loss: {:.1f}'
#                       .format(epoch, valEpocws, str(i_val).zfill(3), str(steps_per_epoch_val).zfill(3),
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

        

# class coco2:
#     def __init__(self, annotations_file_path):
#         self.coco = COCO(annotations_file_path)
#         self.categories = self.coco.loadCats(self.coco.getCatIds())
#         self.category_names = [category['name'] for category in self.categories]
#         self.category_ids = self.coco.getCatIds()
#         self.image_ids = self.coco.getImgIds()
#         self.annotations = self.coco.loadAnns(self.coco.getAnnIds())
#         self.ids = list(self.coco.imgToAnns.keys())     ## initializes list of image ids
#         self.cat2cat = dict()                       ## to map category ids to integer indices, only for internal usage
        
#         for cat in self.coco.cats.keys():
#             self.cat2cat[cat] = len(self.cat2cat)
        
#         self.classfreq = {category_name: 0 for category_name in self.category_names}

#         self.get_class_frequencies()    ## will initialize the class frequencies for the dataset

#     def get_class_frequencies(self):
#             for annotation in self.annotations:
#                 category_id = annotation['category_id']
#                 category_name = self.coco.loadCats(category_id)[0]['name']
#                 self.classfreq[category_name] += 1

#     def print_statistics(self):
#             total_images = len(self.image_ids)
#             total_annotations = len(self.annotations)
#             average_annotations_per_image = total_annotations / total_images
#             print("Total images:", total_images)
#             print("Total annotations:", total_annotations)
#             print("Average annotations per image:", average_annotations_per_image)

#             # for key in self.classfreq:
#             #     print(key, self.classfreq[key])
#             print(self.cat2cat)




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

# torch.cuda.empty_cache()
            ## saving checkpoints
            # if i % 100 == 0:
                
            #     torch.save(self.model_val.state_dict(), os.path.join(
            #         'models/{}/{}/models_val/'.format(self.type, self.model_name), 'model-{}-{}.ckpt'.format(epoch + 1, i + 1)))
            #     torch.save(self.weights, os.path.join(
            #         'models/{}/{}/weights/'.format(self.type, self.model_name), 'weights-{}-{}.ckpt'.format(epoch + 1, i + 1)))  


            ## ema model update
            # self.model_train_ema.update(self.model_train)

            # print("-----------training loop weights after step-------------\n")
            # for name, p in self.model_train.named_parameters():
                # print(p)
                # break


# def validate_multi(val_loader, model, ema_model, printloss = False, weighted = False, sum = True, criteria = BCEloss(), weights = torch.ones(80)):
#     print("starting validation")
#     Sig = torch.nn.Sigmoid()
#     preds_regular = []
#     preds_ema = []
#     targets = []
#     for i, (input, target) in enumerate(val_loader):
#         target = target
#         target = target.max(dim=1)[0]
#         # compute output
#         with torch.no_grad():
#             with autocast():
#                 ## to print loss
#                 if printloss and i%100 == 0:
#                     output_reg = model(input.cuda()).float()
#                     loss = criteria(output_reg, target.cuda(), weights.cuda())


#                 output_regular = Sig(model(input.cuda())).cpu()
#                 output_ema = Sig(ema_model.module(input.cuda())).cpu()

#         # for mAP calculation
#         preds_regular.append(output_regular.cpu().detach())
#         preds_ema.append(output_ema.cpu().detach())
#         targets.append(target.cpu().detach())

#     mAP_score_regular = mAP(torch.cat(targets).numpy(), torch.cat(preds_regular).numpy())
#     mAP_score_ema = mAP(torch.cat(targets).numpy(), torch.cat(preds_ema).numpy())
#     print("mAP score regular {:.2f}, mAP score EMA {:.2f}".format(mAP_score_regular, mAP_score_ema))
#     return max(mAP_score_regular, mAP_score_ema)




# l = lines[i].split(' ')
#             # print(l)
#             first_tensor = float(l[7].split('[')[1][:-1])
#             remaining_tensor = l[8:]
#             #print(remaining_tensor)
#             for rem_tensor in range(len(remaining_tensor)):
#                 remaining_tensor[rem_tensor] = float(remaining_tensor[rem_tensor].split(',')[0])
#             remaining_tensor.insert(0,first_tensor)
#             # print(remaining_tensor)
#             for p in range(1,ind):
#                 rem_tens = (lines[i+p].split(','))
#                 rem_tens = rem_tens[0:-1]
#                 #print(rem_tens)
#                 for rem_tensor_val in range(len(rem_tens)):
#                     rem_tens[rem_tensor_val] = float(rem_tens[rem_tensor_val])
#                 #print(rem_tens)
#                 remaining_tensor.extend(rem_tens)
#             # print(remaining_tensor)
#             last_line = lines[i+ind].split(',')
#             last_line = last_line[0:-1]
#             #first_tensor_list = float(last_line[-1].split(']')[0])
#             last_num = float(last_line[-1].split(']')[0])
#             last_few_nums = last_line[0:-1]
#             for rem_tensor_val in range(len(last_few_nums)):
#                 last_few_nums[rem_tensor_val] = float(last_few_nums[rem_tensor_val])
#             #print(rem_tens)
#             last_few_nums.append(last_num)
#             #print(last_few_nums)
#             remaining_tensor.extend(last_few_nums)
#             # print(remaining_tensor)
#             unweighted_val_loss_vectors.append(remaining_tensor)
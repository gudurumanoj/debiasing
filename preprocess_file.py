import numpy as np
import matplotlib.pyplot as plt
import os
from distfit import distfit


from cocostats import coco2014

import seaborn as sns
import pandas as pd

import ast


def getcocofreq():
    path = '/raid/ganesh/prateekch/MSCOCO_2014'
    train_annotation_file_path = os.path.join(path, 'annotations/instances_train2014.json')
    val_annotation_file_path = os.path.join(path, 'annotations/instances_val2014.json')

    coco2014_train = coco2014(train_annotation_file_path)
    coco2014_val = coco2014(val_annotation_file_path)

    return coco2014_train, coco2014_val



def find_index(lines, string = "device='cuda:0'"):
    for i in range(len(lines)):
        if string in lines[i]:
            return i


def return_tensor(lines, ind, i=0):
    l = (lines[i].strip().split(','))
    # print("----------------------------")
    # print(l)
    first_tensor = float(l[0].split('[')[1])
    remaining_tensor = [float(x) for x in l[1:-1] if x]
    # print(remaining_tensor)
    # remaining_tensor
    remaining_tensor.insert(0,first_tensor)
    # print(remaining_tensor)
    for p in range(1,ind):
        rem_tens = (lines[i+p].strip().split(','))
        rem_tens = rem_tens[0:-1]
        # print(rem_tens)
        for rem_tensor_val in range(len(rem_tens)):
            rem_tens[rem_tensor_val] = float(rem_tens[rem_tensor_val][:-1])
        #print(rem_tens)
        remaining_tensor.extend(rem_tens)
    # print(remaining_tensor)
    # print(lines[i+ind])
    if 'grad_fn' in lines[i+ind]:
        last_line = lines[i+ind].strip().split(',')
        last_line = last_line[0:-2]
    else:
        last_line = lines[i+ind].strip().split(',')
        if "device='cuda:0'" in last_line[0:-1] or " device='cuda:0'" in last_line[0:-1]:
            if "device='cuda:0'" in last_line[0:-2] or " device='cuda:0'" in last_line[0:-2]:
                last_line = last_line[0:-3]
            else:
                last_line = last_line[0:-2]
        else:
            last_line = last_line[0:-1]
            # print("insdie else",last_line)
            
    if len(last_line):
        last_num = float(last_line[-1].split(']')[0])
        last_few_nums = last_line[0:-1]
        for rem_tensor_val in range(len(last_few_nums)):
            last_few_nums[rem_tensor_val] = float(last_few_nums[rem_tensor_val])
        #print(rem_tens)
        last_few_nums.append(last_num)
        #print(last_few_nums)
        remaining_tensor.extend(last_few_nums)

    # print(len(remaining_tensor))
    # print(remaining_tensor)

    return remaining_tensor


def return_acc(lines,ind,i=0):
    l = lines[i].strip().split(' ')
    # print(l)
    first_tensor = float(l[3].split('[')[1])
    remaining_tensor = l[4:]
    remaining_tensor[:] = [x for x in remaining_tensor if x]
    for rem_tensor_val in range(len(remaining_tensor)):
        remaining_tensor[rem_tensor_val] = float(remaining_tensor[rem_tensor_val])
    remaining_tensor.insert(0,first_tensor)
    # print(remaining_tensor)
    for p in range(1,ind):
        rem_tens = (lines[i+p].strip().split(' '))
        # rem_tens = rem_tens[0:-1]
        rem_tens[:] = [x for x in rem_tens if x]
        # print(rem_tens)
        for rem_tensor_val in range(len(rem_tens)):
            rem_tens[rem_tensor_val] = float(rem_tens[rem_tensor_val])
        # print(rem_tens)
        remaining_tensor.extend(rem_tens)
        
    last_line = lines[i+ind].strip().split(' ')
    last_line[:] = [x for x in last_line if x and x != ']']
    # print(last_line)
    last_num = float(last_line[-1].split(']')[0])
    last_few_nums = last_line[0:-1]
    # last_few_nums[:] = [x for x in last_few_nums if (x and x != ']')]
    for rem_tensor_val in range(len(last_few_nums)):
        last_few_nums[rem_tensor_val] = float(last_few_nums[rem_tensor_val])
    last_few_nums.append(last_num)
    remaining_tensor.extend(last_few_nums)

    # print(len(remaining_tensor))
    return remaining_tensor
    # for i in range(len(lines)):

                                             

def convert_to_df(file_name, name = ''):
    """
    Function to convert the txt file to a dataframe
    :param file_name: name of the txt file
    :return: dataframe
    text file has the following format:
        - valEpocw [0/80], Step [000/642], LR 4.0e-06, Loss: 4001.5 (has 6 lines of this format for each valEpocw and step)
        - mAP score regular 0.0000, map score ema 0.0000
    """
    ## convert txt to csv
    print(name)
    with open(file_name, 'r') as f:
        lines = f.readlines()
    # map_data = []
    train_mAP = []
    val_mAP = []
    unweighted_val_loss = []
    summed_Up_meta_val_loss = []
    weighted_val_loss= []
    main_model_weighted_train_loss =  []
    weights = []

    unweighted_val_loss_vectors = []
    summed_Up_meta_val_loss_vectors = []
    weighted_val_loss_vectors = []
    main_model_weighted_train_loss_vectors = []

    train_acc1 = []
    train_acc2 = []
    train_map_ = []

    confusion1_dict = dict()
    confusion2_dict = dict()

    for i in range(80):
        confusion1_dict[i] = []
        confusion2_dict[i] = []

    for i in range(len(lines)):
        if 'Train_data_mAP' in lines[i]:
            # print(lines[i].split(' '))
            train_mAP.append(float(lines[i].split(' ')[3][:-1]))
            # data.append(lines[i].split(' ')[-2:])
        elif 'Val_data_mAP' in lines[i]:
            # print(lines[i].split(' '))
            val_mAP.append(float(lines[i].split(' ')[3][:-1]))
            # data.append(lines[i].split(' ')[-2:])

        elif 'Main Model Unweighted Validation Loss' in lines[i]:
            #print(lines[i])
            unweighted_val_loss.append(float(lines[i].split(' ')[-1]))
        elif 'Meta Learning Summed up Validation Loss' in lines[i] or 'Meta Learning Max Validation Loss' in lines[i]:
            summed_Up_meta_val_loss.append(float(lines[i].split(' ')[-1]))
        elif 'Main Model Validation Loss' in lines[i]:
            weighted_val_loss.append(float(lines[i].split(' ')[-1]))
        elif 'Main Model Weighted Training Loss' in lines[i] or 'Epoch' in lines[i]:
            main_model_weighted_train_loss.append(float(lines[i].split(' ')[-1]))

        elif "{} Main Model Unweighted Val Loss:".format(name) in lines[i]:
            # print('----------here----------')
            if i+20 < len(lines):
                ind = find_index(lines[i:i+20])
            
            unweighted_val_loss_vectors.append(return_tensor(lines[i:i+ind+1], ind))
        
        elif "{} Main Model Weighted Val Loss:".format(name) in lines[i]:
            # print('----------here----------')
            if i+20 < len(lines):
                ind = find_index(lines[i:i+20])
            
            # print(remaining_tensor)
            weighted_val_loss_vectors.append(return_tensor(lines[i:i+ind+1], ind))

        elif "{} Meta Model:".format(name) in lines[i]:
            # print('----------here----------')
            # print(i)
            if i+20 < len(lines):
                # print(lines[i:i+20])
                ind = find_index(lines[i:i+20])
            summed_Up_meta_val_loss_vectors.append(return_tensor(lines[i:i+ind+1], ind))
        
        elif "{} Train Loss:".format(name[:3]) in lines[i]:
            # print(i)
            # print('----------here----------')
            if i+20 < len(lines):
                ind = find_index(lines[i:i+20])
            
            main_model_weighted_train_loss_vectors.append(return_tensor(lines[i:i+ind+1], ind))


        elif "Accuracy th:0.5" in lines[i]:
            # print(lines[i].split(' '))
            if i+15 < len(lines):
                ind = find_index(lines[i:i+15],"]")
            train_acc1.append(return_acc(lines[i:i+ind+1],ind))
            # data.append(lines[i].split(' ')[-2:])

        elif "Accuracy th:0.7" in lines[i]:
            # print(lines[i].split(' '))
            if i+15 < len(lines):
                ind = find_index(lines[i:i+15],"]")
            train_acc2.append(return_acc(lines[i:i+ind+1],ind))
            # data.append(lines[i].split(' ')[-2:])
        
        elif "Avg Prec:" in lines[i]:
            # print(lines[i].split(' '))
            if i+15 < len(lines):
                ind = find_index(lines[i:i+15],"]")
            train_map_.append(return_acc(lines[i:i+ind+1],ind))
            # data.append(lines[i].split(' ')[-2:])

        elif "Confusion stats:0.5" in lines[i]:
            # print(lines[i].split(' '))
            
            confusion1 = ast.literal_eval(lines[i][22:])
            for key in confusion1:
                confusion1_dict[key].append(confusion1[key])


        elif "Confusion stats:0.7" in lines[i]:
            # print(lines[i].split(' '))
            confusion2 = ast.literal_eval(lines[i][22:])
            for key in confusion2:
                confusion2_dict[key].append(confusion2[key])

        elif 'tensor' in lines[i]:
            l = (lines[i].split(','))
            first_tensor = float(l[0].split('[')[1])
            remaining_tensor = l[1:-1]
            #print(remaining_tensor)
            for rem_tensor in range(len(remaining_tensor)):
                remaining_tensor[rem_tensor] = float(remaining_tensor[rem_tensor])
            remaining_tensor.insert(0,first_tensor)
            #print(remaining_tensor)
            if (i+13 < len(lines)) and ("device='cuda:0')" in lines[i+13]):
                # print("=====================================\n")
                for p in range(1,13):
                    rem_tens = (lines[i+p].split(','))
                    # print(rem_tens)
                    rem_tens = rem_tens[0:-1]
                    #print(rem_tens)
                    for rem_tensor_val in range(len(rem_tens)):
                        rem_tens[rem_tensor_val] = float(rem_tens[rem_tensor_val])
                    #print(rem_tens)
                    remaining_tensor.extend(rem_tens)

                last_line = lines[i+13].split(',')
                last_line = last_line[0:-1]
                #first_tensor_list = float(last_line[-1].split(']')[0])
                last_num = float(last_line[-1].split(']')[0])
                last_few_nums = last_line[0:-1]
                for rem_tensor_val in range(len(last_few_nums)):
                    last_few_nums[rem_tensor_val] = float(last_few_nums[rem_tensor_val])
                #print(rem_tens)
                last_few_nums.append(last_num)
                #print(last_few_nums)
                remaining_tensor.extend(last_few_nums)
                #print(remaining_tensor)
                weights.append(remaining_tensor)
            else:
                # print("=====================================\n")
                for p in range(1,8):
                    rem_tens = (lines[i+p].split(','))
                    rem_tens = rem_tens[0:-1]
                    #print(rem_tens)
                    for rem_tensor_val in range(len(rem_tens)):
                        rem_tens[rem_tensor_val] = float(rem_tens[rem_tensor_val])
                    #print(rem_tens)
                    remaining_tensor.extend(rem_tens)
                last_line = lines[i+8].split(',')
                last_line = last_line[0:-1]
                #first_tensor_list = float(last_line[-1].split(']')[0])

                last_num = float(last_line[-1].split(']')[0])
                last_few_nums = last_line[0:-1]
                for rem_tensor_val in range(len(last_few_nums)):
                    last_few_nums[rem_tensor_val] = float(last_few_nums[rem_tensor_val])
                #print(rem_tens)
                last_few_nums.append(last_num)
                #print(last_few_nums)
                remaining_tensor.extend(last_few_nums)
                #print("=====================================\n")

                #print(remaining_tensor)
                #print(remaining_tensor)
                weights.append(remaining_tensor)
            #print("\n\n")
            
    avg_unweighted_val_loss = []
    k= 0 
    while(k< len(unweighted_val_loss)):
        avg_unweighted_val_loss.append(np.average(unweighted_val_loss[k:k+4]))
        k += 4
        
    avg_summed_Up_meta_val_loss = []
    k= 0 
    while(k< len(summed_Up_meta_val_loss)):
        avg_summed_Up_meta_val_loss.append(np.average(summed_Up_meta_val_loss[k:k+4]))
        k +=4
       
    avg_main_model_val_loss = [] 
    k= 0 
    while(k< len(weighted_val_loss)):
        avg_main_model_val_loss.append(np.average(weighted_val_loss[k:k+4]))
        k +=4
        
    avg_main_model_weighted_train_loss = []
    k= 0 
    while(k< len(main_model_weighted_train_loss)):
        avg_main_model_weighted_train_loss.append(np.average(main_model_weighted_train_loss[k:k+7]))
        k +=7

    unweighted_val_loss_vectors = np.array(unweighted_val_loss_vectors)
    summed_Up_meta_val_loss_vectors = np.array(summed_Up_meta_val_loss_vectors)
    weighted_val_loss_vectors = np.array(weighted_val_loss_vectors)
    main_model_weighted_train_loss_vectors = np.array(main_model_weighted_train_loss_vectors)

    # train_acc1 = np.array(train_acc1)
    # train_acc2 = np.array(train_acc2)
    # train_map_ = np.array(train_map_)

    avg_unweighted_val_loss_vectors = []
    k= 0
    while(k< unweighted_val_loss_vectors.shape[0]):
        avg_unweighted_val_loss_vectors.append(np.mean(unweighted_val_loss_vectors[k:k+4],axis=0))
        k += 4
    # print("Unweighted Val Loss Vectors", len(avg_unweighted_val_loss_vectors))

    avg_summed_Up_meta_val_loss_vectors = []
    k= 0
    while(k< summed_Up_meta_val_loss_vectors.shape[0]):
        avg_summed_Up_meta_val_loss_vectors.append(np.mean(summed_Up_meta_val_loss_vectors[k:k+4],axis=0))
        k += 4
    # print("Summed Up Meta Val Loss Vectors", len(avg_summed_Up_meta_val_loss_vectors))

    avg_weighted_val_loss_vectors = []
    k= 0
    while(k< weighted_val_loss_vectors.shape[0]):
        avg_weighted_val_loss_vectors.append(np.mean(weighted_val_loss_vectors[k:k+4],axis=0))
        k += 4
    # print("Weighted Val Loss Vectors", len(avg_weighted_val_loss_vectors))

    avg_main_model_weighted_train_loss_vectors = []
    k= 0
    while(k< main_model_weighted_train_loss_vectors.shape[0]):
        avg_main_model_weighted_train_loss_vectors.append(np.mean(main_model_weighted_train_loss_vectors[k:k+7],axis=0))
        k += 7
    # print("Main Model Weighted Train Loss Vectors", len(avg_main_model_weighted_train_loss_vectors))
        

    avg_weighted_val_loss_vectors = np.array(avg_weighted_val_loss_vectors)
    avg_main_model_weighted_train_loss_vectors = np.array(avg_main_model_weighted_train_loss_vectors)
    avg_unweighted_val_loss_vectors = np.array(avg_unweighted_val_loss_vectors)
    avg_summed_Up_meta_val_loss_vectors = np.array(avg_summed_Up_meta_val_loss_vectors)

    train_acc_1 = np.array([ele for idx, ele in enumerate(train_acc1) if idx % 4 == 0])
    val_acc_1 = np.array([ele for idx, ele in enumerate(train_acc1) if idx % 4 == 2])
    train_acc_2 = np.array([ele for idx, ele in enumerate(train_acc2) if idx % 4 == 0])
    val_acc_2 = np.array([ele for idx, ele in enumerate(train_acc2) if idx % 4 == 2])

    prec_train = np.array([ele for idx, ele in enumerate(train_map_) if idx % 4 == 0])
    prec_val = np.array([ele for idx, ele in enumerate(train_map_) if idx % 4 == 2])

    conf1_train = dict()
    conf1_val = dict()
    conf2_train = dict()
    conf2_val = dict()

    for i in range(80):
        conf1_train[i] = []
        conf1_val[i] = []
        conf2_train[i] = []
        conf2_val[i] = []

        for j in range(len(confusion1_dict[0])):
            if j % 4 == 0:
                conf1_train[i].append(confusion1_dict[i][j])
                conf2_train[i].append(confusion2_dict[i][j])
            elif j % 4 == 2:
                conf1_val[i].append(confusion1_dict[i][j])
                conf2_val[i].append(confusion2_dict[i][j])



        
    # print("Map DAta", map_data)
    
    return avg_unweighted_val_loss,avg_summed_Up_meta_val_loss,avg_main_model_val_loss,avg_main_model_weighted_train_loss,\
        weights, train_mAP, val_mAP, avg_unweighted_val_loss_vectors, avg_summed_Up_meta_val_loss_vectors, avg_weighted_val_loss_vectors,\
        avg_main_model_weighted_train_loss_vectors, train_acc_1, val_acc_1, train_acc_2, val_acc_2, prec_train, prec_val, \
        conf1_train, conf1_val, conf2_train, conf2_val


def convert_to_df_normal(file_name, twovalues = False, name = ''):
    # print("File Name: ", file_name)
    # print("Two Values: ", twovalues)
    with open(file_name, 'r') as f:
        lines = f.readlines()
    map_data = []
    train_map = []
    val_map = []
    val_loss = []
    train_loss = []

    train_loss_vec = []
    val_loss_vec = []

    train_acc1 = []
    train_acc2 = []
    train_map_ = []

    confusion1_dict = dict()
    confusion2_dict = dict()
    
    for i in range(80):
        confusion1_dict[i] = []
        confusion2_dict[i] = []
    

    # print("Length of lines: ", len(lines))
    for i in range(len(lines)):

        if 'mAP score' in lines[i]:
            # print(lines[i].split(' '))
            map_data.append(float(lines[i].split(' ')[3][:-1]))

        elif 'Val loss valEpocw' in lines[i]:
            # print(lines[i])
            val_loss.append(float(lines[i].split(': ')[-1]))

        elif '{} Train Loss'.format(name) in lines[i]:
            if i+15 < len(lines):
                ind = find_index(lines[i:i+15])

            train_loss_vec.append(return_tensor(lines[i:i+ind+1], ind))

        elif '{} Val Loss'.format(name) in lines[i]:
            if i+15 < len(lines):
                ind = find_index(lines[i:i+15])

            val_loss_vec.append(return_tensor(lines[i:i+ind+1], ind))

        elif "Accuracy th:0.5" in lines[i]:
            # print(lines[i].split(' '))
            if i+15 < len(lines):
                ind = find_index(lines[i:i+15],"]")
            train_acc1.append(return_acc(lines[i:i+ind+1],ind))
            # data.append(lines[i].split(' ')[-2:])

        elif "Accuracy th:0.7" in lines[i]:
            # print(lines[i].split(' '))
            if i+15 < len(lines):
                ind = find_index(lines[i:i+15],"]")
            train_acc2.append(return_acc(lines[i:i+ind+1],ind))
            # data.append(lines[i].split(' ')[-2:])

        elif "Confusion stats:0.5" in lines[i]:
            # print(lines[i].split(' '))
            confusion1 = ast.literal_eval(lines[i][22:])
            for key in confusion1:
                confusion1_dict[key].append(confusion1[key])

        elif "Confusion stats:0.7" in lines[i]:
            # print(lines[i].split(' '))
            confusion2 = ast.literal_eval(lines[i][22:])
            for key in confusion2:
                confusion2_dict[key].append(confusion2[key])


        elif "Avg Prec:" in lines[i]:
            # print(lines[i].split(' '))
            if i+15 < len(lines):
                ind = find_index(lines[i:i+15],"]")
            train_map_.append(return_acc(lines[i:i+ind+1],ind))
            # data.append(lines[i].split(' ')[-2:])
        
        elif 'Epoch' in lines[i]:
            #print(lines[i])
            train_loss.append(float(lines[i].split(': ')[-1]))

    print("Val Loss", len(val_loss))

    avg_unweighted_val_loss = []
    k= 0 
    while(k< len(val_loss)):
        avg_unweighted_val_loss.append(np.average(val_loss[k:k+4]))
        k += 4
    avg_main_model_weighted_train_loss = []
    k= 0 
    while(k< len(train_loss)):
        avg_main_model_weighted_train_loss.append(np.average(train_loss[k:k+7]))
        k +=7

    val_loss_vec = np.array(val_loss_vec)
    train_loss_vec = np.array(train_loss_vec)

    avg_val_loss_vectors = []
    k= 0
    while(k< val_loss_vec.shape[0]):
        avg_val_loss_vectors.append(np.mean(val_loss_vec[k:k+4],axis=0))
        k += 4
    # print("Unweighted Val Loss Vectors", len(avg_unweighted_val_loss_vectors))
        
    avg_train_loss_vectors = []
    k= 0
    while(k< train_loss_vec.shape[0]):
        avg_train_loss_vectors.append(np.mean(train_loss_vec[k:k+7],axis=0))
        k += 7
    # print("Unweighted Val Loss Vectors", len(avg_unweighted_val_loss_vectors))
    
    avg_train_loss_vectors = np.array(avg_train_loss_vectors)
    avg_val_loss_vectors = np.array(avg_val_loss_vectors)



    train_acc_1 = np.array([ele for idx, ele in enumerate(train_acc1) if idx % 4 == 0])
    val_acc_1 = np.array([ele for idx, ele in enumerate(train_acc1) if idx % 4 == 2])
    train_acc_2 = np.array([ele for idx, ele in enumerate(train_acc2) if idx % 4 == 0])
    val_acc_2 = np.array([ele for idx, ele in enumerate(train_acc2) if idx % 4 == 2])

    prec_train = np.array([ele for idx, ele in enumerate(train_map_) if idx % 4 == 0])
    prec_val = np.array([ele for idx, ele in enumerate(train_map_) if idx % 4 == 2])

    conf1_train = dict()
    conf1_val = dict()
    conf2_train = dict()
    conf2_val = dict()

    for i in range(80):
        conf1_train[i] = []
        conf1_val[i] = []
        conf2_train[i] = []
        conf2_val[i] = []

        for j in range(len(confusion1_dict[0])):
            if j % 4 == 0:
                conf1_train[i].append(confusion1_dict[i][j])
                conf2_train[i].append(confusion2_dict[i][j])
            elif j % 4 == 2:
                conf1_val[i].append(confusion1_dict[i][j])
                conf2_val[i].append(confusion2_dict[i][j])


    if not twovalues:
        return avg_unweighted_val_loss,avg_main_model_weighted_train_loss,map_data
    
    else:
        train_map = [ele for idx, ele in enumerate(map_data) if idx % 2 == 0]
        val_map = [ele for idx, ele in enumerate(map_data) if idx % 2 != 0]
        return avg_unweighted_val_loss,avg_main_model_weighted_train_loss,train_map, val_map, \
            avg_train_loss_vectors, avg_val_loss_vectors, train_acc_1, val_acc_1, \
            train_acc_2, val_acc_2, prec_train, prec_val, conf1_train, conf1_val, conf2_train, conf2_val


def plotfig(baseline_metric, debiased_metric, savepath, title, xlabel, ylabel,name, type = 'prec', extra = '_tn', valcoco = None):
    plt.figure(figsize=(15,12))
    ax = plt.gca()
    # if type == 'prec':
    #     ax.set_xlim([0, 100])
    #     ax.set_ylim([0, 100])

    ## modify this to get the max of both the lists to place same axis limits

    baseline_max = max(baseline_metric)
    debiased_max = max(debiased_metric)

    maxval = max(baseline_max, debiased_max) + 100
    minval = min(baseline_max, debiased_max)


    ax.set_xlim([0, maxval+50])  ## adding 50 to the max value to give some space
    ax.set_ylim([0, maxval+50])  ## adding 50 to the max value to give some space

    # traincoco, valcoco = getcocofreq()

    freq_dict = valcoco.class_frequencies
    totalimg, totalannotations, avgannotationsperimg = valcoco.print_statistics(return_stats=True)

    if extra == '_fp':
        def forward(x):
            return x**(1/3)
        def inverse(x):
            return x**3
        
        plt.xscale('function', functions=(forward, inverse))
        plt.yscale('function', functions=(forward, inverse))


    elif extra == '_tn':
        def forward(x):
            return x**(6)
        def inverse(x):
            return x**(1/6)

        plt.xscale('function', functions=(forward, inverse))
        plt.yscale('function', functions=(forward, inverse))

    elif extra == '_tp':
        def forward(x):
            return x**(1/5)

        def inverse(x):
            return x**5
        
        plt.xscale('function', functions=(forward, inverse))
        plt.yscale('function', functions=(forward, inverse))

    for i, (x, y) in enumerate(zip(baseline_metric, debiased_metric)):
        plt.scatter(x, y, s = (freq_dict[i]/10))
        plt.text(x, y, str(i), fontsize=15, ha='center', va='bottom')
    # plt.scatter(baseline_metric, debiased_metric)

    plt.plot(np.linspace(0, maxval), np.linspace(0,maxval), color = 'gray', linestyle = '--')
    plt.title(title, fontsize=24)
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.savefig(os.path.join(savepath, name + extra + '_modified' + '.png'))
    plt.close()


def plot_scatter_plot(baseline_metric, debiased_metric, savepath, title, xlabel, ylabel,name, type = 'prec', valcoco = None):


    if type == 'conf':
        ## both baseline_metric and debiased_metric will be dictionaries containing tn, fp, fn, tp
        plt.figure(figsize=(15,12))

        tn_baseline = [baseline_metric[i][-1][0] for i in range(80)]
        tn_debiased = [debiased_metric[i][-1][0] for i in range(80)]

        fp_baseline = [baseline_metric[i][-1][1] for i in range(80)]
        fp_debiased = [debiased_metric[i][-1][1] for i in range(80)]

        fn_baseline = [baseline_metric[i][-1][2] for i in range(80)]
        fn_debiased = [debiased_metric[i][-1][2] for i in range(80)]

        tp_baseline = [baseline_metric[i][-1][3] for i in range(80)]
        tp_debiased = [debiased_metric[i][-1][3] for i in range(80)]


        # for i, (x, y) in enumerate(zip(tn_baseline, tn_debiased)):
        #     plt.scatter(x, y)
        #     plt.text(x, y, str(i), fontsize=15, ha='center', va='bottom')
        
        # plt.title(title, fontsize=24)
        # plt.xlabel(xlabel, fontsize=20)
        # plt.ylabel(ylabel, fontsize=20)
        # plt.savefig(os.path.join(savepath, name + '_tn.png'))
        # plt.close()
        plotfig(tn_baseline, tn_debiased, savepath, "True Negatives [TN]", xlabel, ylabel,name, type = 'conf', extra = '_tn',valcoco= valcoco)
        plotfig(fp_baseline, fp_debiased, savepath, "False Positives [FP]", xlabel, ylabel,name, type = 'conf', extra = '_fp',valcoco= valcoco)
        plotfig(fn_baseline, fn_debiased, savepath, "False Negatives [FN]", xlabel, ylabel,name, type = 'conf', extra = '_fn',valcoco= valcoco)
        plotfig(tp_baseline, tp_debiased, savepath, "True Positives [TP]", xlabel, ylabel,name, type = 'conf', extra = '_tp',valcoco= valcoco)



    else:
        plt.figure(figsize=(15,12))

        ax = plt.gca()
        if type == 'prec':
            ax.set_xlim([0, 100])
            ax.set_ylim([0, 100])

        for i, (x, y) in enumerate(zip(baseline_metric, debiased_metric)):

            plt.scatter(x, y)
            plt.text(x, y, str(i), fontsize=15, ha='center', va='bottom')
        # plt.scatter(baseline_metric, debiased_metric)

        plt.title(title, fontsize=24)
        plt.xlabel(xlabel, fontsize=20)
        plt.ylabel(ylabel, fontsize=20)
        plt.savefig(os.path.join(savepath, name + '.png'))
        plt.close()

def plot_weights(a,savepath):
#     a = returndata()
    # a = returndata2()

    ## plotting for 10 classes at a time for all 80 classes iteratively and all as subplots
    for i in range(0, 80, 10):
        fig, ax = plt.subplots(10, figsize=(10,20), sharex = True, sharey = True)
        ax = ax.ravel()
        for j in range(10):
            ax[j].hist(a[:,i+j])
            ax[j].set_title(f'Class {i+j+1}')
            ax[j].set_ylabel('Weight')
        plt.savefig(os.path.join(savepath, 'weights_' + str(i) + '.png'))

def plotbar(vanillasum, weightedsum, savepath, title, xlabel, ylabel, name, type = 'prec', label = 'Fixed'):
    plt.figure(figsize=(15,9))
    ax = plt.gca()
    # if type == 'prec':
    #     ax.set_ylim([0, 100])

    x = np.arange(80)
    width = 0.35

    ax.bar(x - width/2, vanillasum, width, label='Vanilla')
    ax.bar(x + width/2, weightedsum, width, label=label)

    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.legend()

    plt.savefig(os.path.join(savepath, name + '.png'))
    plt.close()


#########################################
isVal = True                            #
donotShow = True                        #
Showmap = False                         #
weightPlot = False                      #
scatplot = False                         #
barplot = False                         #
accplot = False                         #
plotloss = False                         #
plot80 = False                          #
seed = False                            #
#########################################

sns.set_theme()

avg_unweighted_val_loss_sum,avg_summed_Up_meta_val_loss_sum,avg_main_model_val_loss_sum,avg_main_model_weighted_train_loss_sum,\
weights_sum, train_mAP_sum, val_mAP_sum, avg_unweighted_val_loss_vectors_sum, avg_summed_Up_meta_val_loss_vectors_sum, \
avg_weighted_val_loss_vectors_sum, avg_main_model_weighted_train_loss_vectors_sum, \
train_acc_1_sum, val_acc_1_sum, train_acc_2_sum, val_acc_2_sum, prec_train_sum, prec_val_sum, \
conf1_train_sum, conf1_val_sum, conf2_train_sum, conf2_val_sum = (convert_to_df("rerun_logs/sum_conf.txt"))


print("--------- Successfully loaded Sum ----------")


avg_unweighted_val_loss_max,avg_summed_Up_meta_val_loss_max,avg_main_model_val_loss_max,avg_main_model_weighted_train_loss_max,\
weights_max, train_mAP_max, val_mAP_max, avg_unweighted_val_loss_vectors_max, avg_summed_Up_meta_val_loss_vectors_max, \
avg_weighted_val_loss_vectors_max, avg_main_model_weighted_train_loss_vectors_max, \
train_acc_1_max, val_acc_1_max, train_acc_2_max, val_acc_2_max, prec_train_max, prec_val_max, \
conf1_train_max, conf1_val_max, conf2_train_max, conf2_val_max = (convert_to_df("rerun_logs/max_conf.txt"))


print("--------- Successfully loaded Max ----------")

avg_unweighted_val_loss_fixed, avg_summed_Up_meta_val_loss_fixed, avg_main_model_val_loss_fixed, avg_main_model_weighted_train_loss_fixed,\
weights_fixed, train_mAP_fixed, val_mAP_fixed, avg_unweighted_val_loss_vectors_fixed, avg_summed_Up_meta_val_loss_vectors_fixed, \
avg_weighted_val_loss_vectors_fixed, avg_main_model_weighted_train_loss_vectors_fixed, \
train_acc_1_fixed, val_acc_1_fixed, train_acc_2_fixed, val_acc_2_fixed, prec_train_fixed, prec_val_fixed, \
conf1_train_fixed, conf1_val_fixed, conf2_train_fixed, conf2_val_fixed = (convert_to_df("rerun_logs/fixedw.txt"))



print("--------- Successfully loaded Fixed Weights ----------")

avg_unweighted_val_loss_upweight, avg_summed_Up_meta_val_loss_upweight, avg_main_model_val_loss_upweight, avg_main_model_weighted_train_loss_upweight,\
weights_upweight, train_mAP_upweight, val_mAP_upweight, avg_unweighted_val_loss_vectors_upweight, avg_summed_Up_meta_val_loss_vectors_upweight, \
avg_weighted_val_loss_vectors_upweight, avg_main_model_weighted_train_loss_vectors_upweight, \
train_acc_1_upweight, val_acc_1_upweight, train_acc_2_upweight, val_acc_2_upweight, prec_train_upweight, prec_val_upweight, \
conf1_train_upweight, conf1_val_upweight, conf2_train_upweight, conf2_val_upweight = (convert_to_df("rerun_logs/upweight.txt"))


print("--------- Successfully loaded Upweight ----------")



avg_unweighted_val_loss_tf, avg_summed_Up_meta_val_loss_tf, avg_main_model_val_loss_tf, avg_main_model_weighted_train_loss_tf,\
weights_tf, train_mAP_tf, val_mAP_tf, avg_unweighted_val_loss_vectors_tf, avg_summed_Up_meta_val_loss_vectors_tf, \
avg_weighted_val_loss_vectors_tf, avg_main_model_weighted_train_loss_vectors_tf, \
train_acc_1_tf, val_acc_1_tf, train_acc_2_tf, val_acc_2_tf, prec_train_tf, prec_val_tf, \
conf1_train_tf, conf1_val_tf, conf2_train_tf, conf2_val_tf = (convert_to_df("rerun_logs/paramsbcetrain.txt"))


print("--------- Successfully loaded TF ----------")






avg_unweighted_val_loss_withoutupper, avg_summed_Up_meta_val_loss_withoutupper, avg_main_model_val_loss_withoutupper, avg_main_model_weighted_train_loss_withoutupper,\
weights_withoutupper, train_mAP_withoutupper, val_mAP_withoutupper, avg_unweighted_val_loss_vectors_withoutupper, avg_summed_Up_meta_val_loss_vectors_withoutupper, \
avg_weighted_val_loss_vectors_withoutupper, avg_main_model_weighted_train_loss_vectors_withoutupper, \
train_acc_1_withoutupper, val_acc_1_withoutupper, train_acc_2_withoutupper, val_acc_2_withoutupper, prec_train_withoutupper, prec_val_withoutupper, \
conf1_train_withoutupper, conf1_val_withoutupper, conf2_train_withoutupper, conf2_val_withoutupper = (convert_to_df("rerun_logs/log.txt"))



print("--------- Successfully loaded withoutupper ----------")


avg_unweighted_val_loss_withupper, avg_summed_Up_meta_val_loss_withupper, avg_main_model_val_loss_withupper, avg_main_model_weighted_train_loss_withupper,\
weights_withupper, train_mAP_withupper, val_mAP_withupper, avg_unweighted_val_loss_vectors_withupper, avg_summed_Up_meta_val_loss_vectors_withupper, \
avg_weighted_val_loss_vectors_withupper, avg_main_model_weighted_train_loss_vectors_withupper, \
train_acc_1_withupper, val_acc_1_withupper, train_acc_2_withupper, val_acc_2_withupper, prec_train_withupper, prec_val_withupper, \
conf1_train_withupper, conf1_val_withupper, conf2_train_withupper, conf2_val_withupper = (convert_to_df("rerun_logs/log2.txt"))


print("--------- Successfully loaded withupper ----------")




avg_unweighted_val_loss_bce,avg_main_model_weighted_train_loss_bce,train_map_bce, val_map_bce, \
avg_train_loss_vectors_bce, avg_val_loss_vectors_bce, train_acc_1_bce, val_acc_1_bce, \
train_acc_2_bce, val_acc_2_bce, prec_train_bce, prec_val_bce, \
conf1_train_bce, conf1_val_bce, conf2_train_bce, conf2_val_bce      = convert_to_df_normal("rerun_logs/bce_conf.txt",True)


print("--------- Successfully loaded BCE ----------")

avg_unweighted_val_loss_bce2, avg_main_model_weighted_train_loss_bce2, train_map_bce2, val_map_bce2, \
avg_train_loss_vectors_bce2, avg_val_loss_vectors_bce2, train_acc_1_bce2, val_acc_1_bce2, \
train_acc_2_bce2, val_acc_2_bce2, prec_train_bce2, prec_val_bce2, \
conf1_train_bce2, conf1_val_bce2, conf2_train_bce2, conf2_val_bce2      = convert_to_df_normal("rerun_logs/bceweights.txt",True)


print("--------- Successfully loaded BCE2 ----------")


if seed:
    avg_unweighted_val_loss_bce280, avg_main_model_weighted_train_loss_bce280, train_map_bce280, val_map_bce280, \
    avg_train_loss_vectors_bce280, avg_val_loss_vectors_bce280, train_acc_1_bce280, val_acc_1_bce280, \
    train_acc_2_bce280, val_acc_2_bce280, prec_train_bce280, prec_val_bce280, \
    conf1_train_bce280, conf1_val_bce280, conf2_train_bce280, conf2_val_bce280      = convert_to_df_normal("rerun_logs/bceweights_80.txt",True)

    avg_unweighted_val_loss_sum_0,avg_summed_Up_meta_val_loss_sum_0,avg_main_model_val_loss_sum_0,avg_main_model_weighted_train_loss_sum_0,\
    weights_sum_0, train_mAP_sum_0, val_mAP_sum_0, avg_unweighted_val_loss_vectors_sum_0, avg_summed_Up_meta_val_loss_vectors_sum_0, \
    avg_weighted_val_loss_vectors_sum_0, avg_main_model_weighted_train_loss_vectors_sum_0, \
    train_acc_1_sum_0, val_acc_1_sum_0, train_acc_2_sum_0, val_acc_2_sum_0, prec_train_sum_0, prec_val_sum_0 = (convert_to_df("rerun_logs/seed_0/sum.txt"))


    avg_unweighted_val_loss_sum_0 = np.array(avg_unweighted_val_loss_sum_0)
    avg_summed_Up_meta_val_loss_sum_0 = np.array(avg_summed_Up_meta_val_loss_sum_0)
    avg_main_model_val_loss_sum_0 = np.array(avg_main_model_val_loss_sum_0)
    avg_main_model_weighted_train_loss_sum_0 = np.array(avg_main_model_weighted_train_loss_sum_0)
    weights_sum_0 = np.array(weights_sum_0)
    train_mAP_sum_0 = np.array(train_mAP_sum_0)
    val_mAP_sum_0 = np.array(val_mAP_sum_0)


    print("--------- Successfully loaded Sum Seed 0 ----------")
    # print(avg_unweighted_val_loss_sum_0.shape)

    avg_unweighted_val_loss_sum_1,avg_summed_Up_meta_val_loss_sum_1,avg_main_model_val_loss_sum_1,avg_main_model_weighted_train_loss_sum_1,\
    weights_sum_1, train_mAP_sum_1, val_mAP_sum_1, avg_unweighted_val_loss_vectors_sum_1, avg_summed_Up_meta_val_loss_vectors_sum_1, \
    avg_weighted_val_loss_vectors_sum_1, avg_main_model_weighted_train_loss_vectors_sum_1, \
    train_acc_1_sum_1, val_acc_1_sum_1, train_acc_2_sum_1, val_acc_2_sum_1, prec_train_sum_1, prec_val_sum_1 = (convert_to_df("rerun_logs/seed_256/sum.txt"))

    avg_unweighted_val_loss_sum_1 = np.array(avg_unweighted_val_loss_sum_1)
    avg_summed_Up_meta_val_loss_sum_1 = np.array(avg_summed_Up_meta_val_loss_sum_1)
    avg_main_model_val_loss_sum_1 = np.array(avg_main_model_val_loss_sum_1)
    avg_main_model_weighted_train_loss_sum_1 = np.array(avg_main_model_weighted_train_loss_sum_1)
    weights_sum_1 = np.array(weights_sum_1)
    train_mAP_sum_1 = np.array(train_mAP_sum_1)
    val_mAP_sum_1 = np.array(val_mAP_sum_1)

    print("--------- Successfully loaded Sum Seed 256 ----------")


    avg_unweighted_val_loss_max_0,avg_summed_Up_meta_val_loss_max_0,avg_main_model_val_loss_max_0,avg_main_model_weighted_train_loss_max_0,\
    weights_max_0, train_mAP_max_0, val_mAP_max_0, avg_unweighted_val_loss_vectors_max_0, avg_summed_Up_meta_val_loss_vectors_max_0, \
    avg_weighted_val_loss_vectors_max_0, avg_main_model_weighted_train_loss_vectors_max_0, \
    train_acc_1_max_0, val_acc_1_max_0, train_acc_2_max_0, val_acc_2_max_0, prec_train_max_0, prec_val_max_0 = (convert_to_df("rerun_logs/seed_0/max.txt"))

    avg_unweighted_val_loss_max_0 = np.array(avg_unweighted_val_loss_max_0)
    avg_summed_Up_meta_val_loss_max_0 = np.array(avg_summed_Up_meta_val_loss_max_0)
    avg_main_model_val_loss_max_0 = np.array(avg_main_model_val_loss_max_0)
    avg_main_model_weighted_train_loss_max_0 = np.array(avg_main_model_weighted_train_loss_max_0)
    weights_max_0 = np.array(weights_max_0)
    train_mAP_max_0 = np.array(train_mAP_max_0)
    val_mAP_max_0 = np.array(val_mAP_max_0)

    print("--------- Successfully loaded Max Seed 0 ----------")

    avg_unweighted_val_loss_max_1,avg_summed_Up_meta_val_loss_max_1,avg_main_model_val_loss_max_1,avg_main_model_weighted_train_loss_max_1,\
    weights_max_1, train_mAP_max_1, val_mAP_max_1, avg_unweighted_val_loss_vectors_max_1, avg_summed_Up_meta_val_loss_vectors_max_1, \
    avg_weighted_val_loss_vectors_max_1, avg_main_model_weighted_train_loss_vectors_max_1, \
    train_acc_1_max_1, val_acc_1_max_1, train_acc_2_max_1, val_acc_2_max_1, prec_train_max_1, prec_val_max_1 = (convert_to_df("rerun_logs/seed_256/max.txt"))

    avg_unweighted_val_loss_max_1 = np.array(avg_unweighted_val_loss_max_1)
    avg_summed_Up_meta_val_loss_max_1 = np.array(avg_summed_Up_meta_val_loss_max_1)
    avg_main_model_val_loss_max_1 = np.array(avg_main_model_val_loss_max_1)
    avg_main_model_weighted_train_loss_max_1 = np.array(avg_main_model_weighted_train_loss_max_1)
    weights_max_1 = np.array(weights_max_1)
    train_mAP_max_1 = np.array(train_mAP_max_1)
    val_mAP_max_1 = np.array(val_mAP_max_1)

    print("--------- Successfully loaded Max Seed 256 ----------")


if barplot:
    plotbar(prec_val_sum[-1,:], prec_val_fixed[-1,:], "rerun_logs/plots", "Validation Precision [Fixed v Vanilla]", "Classes", "Precision", "prec_val_sum_vs_fixed")
    plotbar(prec_val_sum[-1,:], prec_val_upweight[-1,:], "rerun_logs/plots", "Validation Precision [Upeighted v Vanilla]", "Classes", "Precision", "prec_val_sum_vs_upweight", label = 'Upweighted')
    
    ## barplot of weights
    plotbar(weights_sum[-1], weights_upweight[-1], "rerun_logs/plots", "Weights [Upweighted v Vanilla]", "Classes", "Weights", "weights_sum_vs_upweight", label = 'Upweighted')

    weights_sum_arr = np.array(weights_sum[-1])
    weights_upweight_arr = np.array(weights_upweight[-1])

    plt.bar(np.arange(80), weights_sum_arr - weights_upweight_arr)
    plt.yticks([-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1])
    plt.title("Difference in Weights [Vanilla - Upweighted]")
    plt.xlabel("Classes")
    plt.ylabel("Difference")
    plt.savefig("rerun_logs/plots/weights_diff_sum_upweight.png")
    plt.close()


if plot80:
    plt.figure(figsize=(10,6))
    plt.plot([p for p in range(len(val_map_bce80))], val_map_bce80,label='BCE Validation', marker='v')
    plt.plot([p for p in range(len(val_map_bce280))], val_map_bce280,label='BCE Static Reweighted', marker='o')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('mAP')
    plt.title('Validation mAP')
    plt.savefig('rerun_logs/plots/val_mAP_bce80.png')
    plt.close()

    plt.figure(figsize=(10,6))
    plt.plot([p for p in range(len(train_map_bce80))], train_map_bce80,label='BCE Training', marker='v')
    plt.plot([p for p in range(len(train_map_bce280))], train_map_bce280,label='BCE Static Reweighted Training', marker='o')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('mAP')
    plt.title('Training mAP')
    plt.savefig('rerun_logs/plots/train_mAP_bce80.png')
    plt.close()




if accplot:
    plt.figure(figsize=(10,6))
    plt.plot([p for p in range(len(val_mAP_sum))], val_mAP_sum,label='Sum Objective', marker='v')
    plt.plot([p for p in range(len(val_mAP_max))], val_mAP_max,label='Max Objective', marker='o')
    plt.plot([p for p in range(len(val_mAP_fixed))], val_mAP_fixed,label='Fixed weights', marker='x')
    plt.plot([p for p in range(len(val_map_bce))], val_map_bce,label='BCE', marker='>')
    plt.plot([p for p in range(len(val_mAP_upweight))], val_mAP_upweight,label='Upweighted Training', marker='d')
    plt.plot([p for p in range(len(val_map_bce2))], val_map_bce2,label='Weighted BCE with lambdas learnt', marker='s')
    # plt.plot([p for p in range(len(val_mAP_tf))], val_mAP_tf,label='With BCE Weights', marker='p')
    plt.plot([p for p in range(len(val_mAP_tf2))], val_mAP_tf2,label='With BCE Weights', marker='p')
    plt.plot([p for p in range(len(val_mAP_withoutupper))], val_mAP_withoutupper,label='Without upper limit on NP-set', marker='.')
    plt.plot([p for p in range(len(val_mAP_withupper))], val_mAP_withupper,label='With upper limit on NP-set', marker='<')
    

    # plt.legend(loc='upper right')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('mAP')
    plt.title('Validation mAP')
    plt.savefig('rerun_logs/plots/val_mAP_3.png')

    plt.close()

    plt.figure(figsize=(10,6))
    plt.plot([p for p in range(len(train_mAP_sum))], train_mAP_sum,label='Sum Objective', marker='v')
    plt.plot([p for p in range(len(train_mAP_max))], train_mAP_max,label='Max Objective', marker='o')
    plt.plot([p for p in range(len(train_mAP_fixed))], train_mAP_fixed,label='Fixed Weights', marker='x')
    plt.plot([p for p in range(len(train_map_bce))], train_map_bce,label='BCE', marker='>')
    plt.plot([p for p in range(len(train_mAP_upweight))], train_mAP_upweight,label='Upweighted Training', marker='d')
    plt.plot([p for p in range(len(train_map_bce2))], train_map_bce2,label='Weighted BCE with lambdas learnt', marker='s')
    # plt.plot([p for p in range(len(train_mAP_tf))], train_mAP_tf,label='With BCE Weights', marker='p')
    plt.plot([p for p in range(len(train_mAP_tf2))], train_mAP_tf2,label='With BCE Weights', marker='p')
    plt.plot([p for p in range(len(train_mAP_withoutupper))], train_mAP_withoutupper,label='Without upper limit on NP-set', marker='.')
    plt.plot([p for p in range(len(train_mAP_withupper))], train_mAP_withupper,label='With upper limit on NP-set', marker='<')
    # plt.legend(loc='upper right')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('mAP')
    plt.title('Training mAP')
    plt.savefig('rerun_logs/plots/train_mAP_3.png')

    plt.close()
    

if scatplot:



    train_stats, val_stats = getcocofreq()

    # plot_scatter_plot(prec_train_bce[-1,:], prec_val_bce[-1,:], "rerun_logs/plots", "Precision BCE", "Train", "Val", "val_vs_train_prec_bce", type = 'prec')
    # plot_scatter_plot(prec_train_sum[-1,:], prec_val_sum[-1,:], "rerun_logs/plots", "Precision SUM", "Train", "Val", "val_vs_train_prec_sum", type = 'prec')


    # plot_scatter_plot(conf1_val_bce, conf1_val_sum, "rerun_logs/plots", "Confusion", "BCE", "SUM", "sum_vs_bce_valconf", type = 'conf', valcoco = val_stats)
    # plot_scatter_plot(conf1_val_bce, conf1_val_max, "rerun_logs/plots", "Confusion", "BCE", "MAX", "max_vs_bce_valconf", type = 'conf', valcoco = val_stats)
    plot_scatter_plot(conf1_val_bce, conf1_val_fixed, "rerun_logs/plots/scatterplots", "Confusion", "BCE", "Fixed with Inv Weights", "fixed_vs_bce_valconf", type = 'conf', valcoco = val_stats)
    plot_scatter_plot(conf1_val_bce, conf1_val_upweight, "rerun_logs/plots/scatterplots", "Confusion", "BCE", "Upweighted Trained Learning Lambdas", "upweighted_vs_bce_valconf", type = 'conf', valcoco = val_stats)
    plot_scatter_plot(conf1_val_bce, conf1_val_bce2, "rerun_logs/plots/scatterplots", "Confusion", "BCE", "Static Reweighted BCE", "weighted_vs_bce_valconf", type = 'conf', valcoco = val_stats)

    # plot_scatter_plot(conf1_train_bce, conf1_train_sum, "rerun_logs/plots", "Confusion", "BCE", "SUM", "sum_vs_bce_trainconf", type = 'conf', valcoco = train_stats)
    # plot_scatter_plot(conf1_train_bce, conf1_train_max, "rerun_logs/plots", "Confusion", "BCE", "MAX", "max_vs_bce_trainconf", type = 'conf', valcoco = train_stats)
    plot_scatter_plot(conf1_train_bce, conf1_train_fixed, "rerun_logs/plots/scatterplots", "Confusion", "BCE", "Fixed", "fixed_vs_bce_trainconf", type = 'conf', valcoco = train_stats)
    plot_scatter_plot(conf1_train_bce, conf1_train_upweight, "rerun_logs/plots/scatterplots", "Confusion", "BCE", "Upweighted Trained Learning Lambdas", "upweighted_vs_bce_trainconf", type = 'conf', valcoco = train_stats)
    plot_scatter_plot(conf1_train_bce, conf1_train_bce2, "rerun_logs/plots/scatterplots", "Confusion", "BCE", "Static Reweighted BCE", "weighted_vs_bce_trainconf", type = 'conf', valcoco = train_stats)

## size if 164 because model_train - train dataset, model_train_ema - train dataset, model_train - val dataset, model_train_ema - val dataset

if plotloss:
    ## plotting the loss curves for fixed, bce2, upweighted
    plt.figure(figsize=(10,6))
    # plt.plot([p for p in range(len(avg_main_model_weighted_train_loss_fixed))], avg_main_model_weighted_train_loss_fixed,label='Fixed INV Weights', marker='v')
    plt.plot([p for p in range(len(avg_main_model_weighted_train_loss_bce2))], avg_main_model_weighted_train_loss_bce2,label='Static Reweighted', marker='o')
    plt.plot([p for p in range(len(avg_main_model_weighted_train_loss_upweight))], avg_main_model_weighted_train_loss_upweight,label='Upweighted Trained Learning Lambdas', marker='x')

    plt.title('Training Loss Curves')
    plt.ylabel("Training Loss magnitude")
    plt.xlabel("Epochs")
    plt.legend()
    # plt.savefig('rerun_logs/plots/train_loss_fixed_bce2_upweight.png')
    plt.savefig('rerun_logs/plots/train_loss_bce2_upweight.png')

    plt.close()

    plt.figure(figsize=(10,6))
    # plt.plot([p for p in range(len(avg_unweighted_val_loss_fixed))], avg_unweighted_val_loss_fixed,label='Fixed INV Weights', marker='v')
    plt.plot([p for p in range(len(avg_unweighted_val_loss_bce2))], avg_unweighted_val_loss_bce2,label='Static Reweighted', marker='o')
    plt.plot([p for p in range(len(avg_unweighted_val_loss_upweight))], avg_unweighted_val_loss_upweight,label='Upweighted Trained Learning Lambdas', marker='x')

    plt.title('Validation Loss Curves')
    plt.ylabel("Validation Loss magnitude")
    plt.xlabel("Epochs")
    plt.legend()
    # plt.savefig('rerun_logs/plots/val_loss_fixed_bce2_upweight.png')
    plt.savefig('rerun_logs/plots/val_loss_bce2_upweight.png')
    plt.close()


if(isVal == True and donotShow == False):
    plt.figure(figsize=(10,6))

    plt.plot(avg_unweighted_val_loss_sum, label='Seed 42', marker='X')
    plt.plot(avg_unweighted_val_loss_sum_0, label='Seed 0', marker='v')
    plt.plot(avg_unweighted_val_loss_sum_1, label='Seed 256', marker='*')
    plt.title('Unweighted Validation Loss for Sum Objective')
    plt.ylabel("Validation Loss magnitude")
    plt.xlabel("Epochs")
    plt.legend(loc='upper right')
    plt.savefig('unweighted_main_model_validation_loss_sum.png')
    plt.close()

    plt.plot(avg_main_model_weighted_train_loss_sum, label='Seed 42', marker='X')
    plt.plot(avg_main_model_weighted_train_loss_sum_0, label='Seed 0', marker='v')
    plt.plot(avg_main_model_weighted_train_loss_sum_1, label='Seed 256', marker='*')
    plt.title('Main Model Weighted Training Loss Sum Objective')
    plt.ylabel("Training Loss magnitude")
    plt.xlabel("Epochs")
    plt.legend(loc='upper right')
    plt.savefig('weighted_main_model_train_loss_sum.png')
    plt.close()

    plt.plot(avg_unweighted_val_loss_max, label='Seed 42', marker='X')
    plt.plot(avg_unweighted_val_loss_max_0, label='Seed 0', marker='v')
    plt.plot(avg_unweighted_val_loss_max_1, label='Seed 256', marker='*')
    plt.title('Unweighted Validation Loss for Max Objective')
    plt.ylabel("Validation Loss magnitude")
    plt.xlabel("Epochs")
    plt.legend(loc='upper right')
    plt.savefig('unweighted_main_model_validation_loss_max.png')
    plt.close()

    plt.plot(avg_main_model_weighted_train_loss_max, label='Seed 42', marker='X')
    plt.plot(avg_main_model_weighted_train_loss_max_0, label='Seed 0', marker='v')
    plt.plot(avg_main_model_weighted_train_loss_max_1, label='Seed 256', marker='*')
    plt.title('Main Model Weighted Training Loss Max Objective')
    plt.ylabel("Training Loss magnitude")
    plt.xlabel("Epochs")
    plt.legend(loc='upper right')
    plt.savefig('weighted_main_model_train_loss_max.png')
    plt.close()

elif(isVal == False and donotShow == False):
    


    plt.figure(figsize=(10,6))

    plt.plot([p for p in range(len(avg_main_model_weighted_train_loss_max))], avg_main_model_weighted_train_loss_max, label='Weighted Training Loss Max Objective',marker='X')









    #plt.plot([p for p in range(len(avg_summed_Up_meta_val_loss_sum))], avg_summed_Up_meta_val_loss_sum, label='Sum Objective Objective Value', marker='8')

    plt.plot([p for p in range(len(avg_main_model_weighted_train_loss_sum))], avg_main_model_weighted_train_loss_sum, label='Weighted Training Loss Sum Objective',marker='D')


    plt.plot([p for p in range(len(train_loss_asl))], train_loss_asl, label='ASL Objective Training Loss', marker='v')
    plt.plot([p for p in range(len(train_loss_bce))], train_loss_bce, label='BCE Objective Training Loss', marker='o')




    plt.title('Training Loss Curves Min Max obj, vs sum obj, vs ASL, vs BCE')


    plt.ylabel("Training Loss magnitude")
    plt.xlabel("Epochs")

    plt.legend(loc='right')
    
if(weightPlot):
    weightarray_sum = np.array(weights_sum)
    weightsarray_max = np.array(weights_max)
    plt.figure(figsize=(24,16))

    sns.set_theme()
    sns.boxplot(data=[d for d in weightarray_sum.T])
    plt.title('Box plot: Weights for Sum Objective')
    plt.xlabel('Class ID')
    plt.ylabel('Weight')
    plt.savefig('weights_sum.png')
    plt.close()

    plt.figure(figsize=(24,16))
    sns.set_theme()
    sns.boxplot(data=[d for d in weightsarray_max.T])
    plt.title('Box plot: Weights for Max Objective')
    plt.xlabel('Class ID')
    plt.ylabel('Weight')
    plt.savefig('weights_max.png')
    plt.close()

    ## plotting weights
    savepath = os.path.join('rerun_logs','seed42')

    plt.show()
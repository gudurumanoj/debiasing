import numpy as np
import matplotlib.pyplot as plt
import os
from distfit import distfit

import seaborn as sns
import pandas as pd


def find_index(lines, string = "device='cuda:0'"):
    for i in range(len(lines)):
        if string in lines[i]:
            return i


def return_tensor(lines, ind, i=0):
    l = (lines[i].split(','))
    # print("----------------------------")
    # print(l)
    first_tensor = float(l[0].split('[')[1])
    remaining_tensor = [float(x) for x in l[1:-1] if x]
    remaining_tensor
    remaining_tensor.insert(0,first_tensor)
    # print(remaining_tensor)
    for p in range(1,ind):
        rem_tens = (lines[i+p].split(','))
        rem_tens = rem_tens[0:-1]
        # print(rem_tens)
        for rem_tensor_val in range(len(rem_tens)):
            rem_tens[rem_tensor_val] = float(rem_tens[rem_tensor_val][:-1])
        #print(rem_tens)
        remaining_tensor.extend(rem_tens)
    # print(remaining_tensor)
    if 'grad_fn' in lines[i+ind]:
        last_line = lines[i+ind].split(',')
        last_line = last_line[0:-2]
    else:
        last_line = lines[i+ind].split(',')
        last_line = last_line[0:-1]
    # last_line = lines[i+ind].split(',')
    # print(last_line)
    # last_line = last_line[0:-1]
    #first_tensor_list = float(last_line[-1].split(']')[0])
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



        
    # print("Map DAta", map_data)
    
    return avg_unweighted_val_loss,avg_summed_Up_meta_val_loss,avg_main_model_val_loss,avg_main_model_weighted_train_loss,\
        weights, train_mAP, val_mAP, avg_unweighted_val_loss_vectors, avg_summed_Up_meta_val_loss_vectors, avg_weighted_val_loss_vectors,\
        avg_main_model_weighted_train_loss_vectors, train_acc_1, val_acc_1, train_acc_2, val_acc_2, prec_train, prec_val


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


    if not twovalues:
        return avg_unweighted_val_loss,avg_main_model_weighted_train_loss,map_data
    
    else:
        train_map = [ele for idx, ele in enumerate(map_data) if idx % 2 == 0]
        val_map = [ele for idx, ele in enumerate(map_data) if idx % 2 != 0]
        return avg_unweighted_val_loss,avg_main_model_weighted_train_loss,train_map, val_map, \
            avg_train_loss_vectors, avg_val_loss_vectors, train_acc_1, val_acc_1, \
            train_acc_2, val_acc_2, prec_train, prec_val


def plot_scatter_plot(baseline_metric, debiased_metric, savepath, title, xlabel, ylabel,name):
    plt.figure(figsize=(15,12))

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

# avg_unweighted_val_loss_max,avg_summed_Up_meta_val_loss_max,avg_main_model_val_loss_max,avg_main_model_weighted_train_loss_max,weights_max,train_map_max, val_map_max = (convert_to_df("rerun_logs/max.txt"))
# print(len(avg_unweighted_val_loss_max),len(avg_summed_Up_meta_val_loss_max),len(avg_main_model_val_loss_max),len(avg_main_model_weighted_train_loss_max),len(weights_max),len(train_map_max), len(val_map_max))
# print("----------------------------------------------")

# avg_unweighted_val_loss_sum,avg_summed_Up_meta_val_loss_sum,avg_main_model_val_loss_sum,avg_main_model_weighted_train_loss_sum,weights_sum,train_map_sum, val_map_sum = (convert_to_df("rerun_logs/sum.txt"))
# print(len(avg_unweighted_val_loss_sum),len(avg_summed_Up_meta_val_loss_sum),len(avg_main_model_val_loss_sum),len(avg_main_model_weighted_train_loss_sum),len(weights_sum),len(train_map_sum), len(val_map_sum))
# print("----------------------------------------------")

# val_loss_asl, train_loss_asl, val_map_asl = convert_to_df_normal("baseline_logs/asl2.txt")
# print(len(val_loss_asl), len(train_loss_asl), len(val_map_asl))
# print("----------------------------------------------")

# val_loss_bce, train_loss_bce, train_map_bce, val_map_bce = convert_to_df_normal("baseline_logs/bce.txt",True)
# print(len(val_loss_bce), len(train_loss_bce), len(train_map_bce), len(val_map_bce))
# print("----------------------------------------------")

# print("Succesfully loaded")

# #print(len(avg_unweighted_val_loss))
# print("Averaged Main Model Unweighted Validation Loss")

avg_unweighted_val_loss_max,avg_summed_Up_meta_val_loss_max,avg_main_model_val_loss_max,avg_main_model_weighted_train_loss_max,\
weights_max, train_mAP_max, val_mAP_max, avg_unweighted_val_loss_vectors_max, avg_summed_Up_meta_val_loss_vectors_max, \
avg_weighted_val_loss_vectors_max, avg_main_model_weighted_train_loss_vectors_max, \
train_acc_1_max, val_acc_1_max, train_acc_2_max, val_acc_2_max, prec_train_max, prec_val_max = (convert_to_df("rerun_logs/max_acc.txt"))



#########################################
isVal = True                            #
donotShow = True                        #
Showmap = False                         #
weightPlot = False                      #
scatplot = True                         #
#########################################

sns.set_theme()


# print(avg_unweighted_val_loss_max.shape)
# print(avg_summed_Up_meta_val_loss_max.shape)
# print(avg_main_model_val_loss_max.shape)
# print(avg_main_model_weighted_train_loss_max.shape)
# print(weights_max.shape)
# print(train_mAP_max.shape)
# print(val_mAP_max.shape)
print(avg_unweighted_val_loss_vectors_max.shape)
print(avg_summed_Up_meta_val_loss_vectors_max.shape)
print(avg_weighted_val_loss_vectors_max.shape)
print(avg_main_model_weighted_train_loss_vectors_max.shape)
print(train_acc_1_max.shape)
print(val_acc_1_max.shape)
print(train_acc_2_max.shape)
print(val_acc_2_max.shape)
print(prec_train_max.shape)
print(prec_val_max.shape)
print("--------- Successfully loaded Max ----------")

avg_unweighted_val_loss_sum,avg_summed_Up_meta_val_loss_sum,avg_main_model_val_loss_sum,avg_main_model_weighted_train_loss_sum,\
weights_sum, train_mAP_sum, val_mAP_sum, avg_unweighted_val_loss_vectors_sum, avg_summed_Up_meta_val_loss_vectors_sum, \
avg_weighted_val_loss_vectors_sum, avg_main_model_weighted_train_loss_vectors_sum, \
train_acc_1_sum, val_acc_1_sum, train_acc_2_sum, val_acc_2_sum, prec_train_sum, prec_val_sum = (convert_to_df("rerun_logs/sum_acc.txt"))

# print(avg_unweighted_val_loss_sum.shape)
# print(avg_summed_Up_meta_val_loss_sum.shape)
# print(avg_main_model_val_loss_sum.shape)
# print(avg_main_model_weighted_train_loss_sum.shape)
# print(weights_sum.shape)
# print(train_mAP_sum.shape)
# print(val_mAP_sum.shape)
print(avg_unweighted_val_loss_vectors_sum.shape)
print(avg_summed_Up_meta_val_loss_vectors_sum.shape)
print(avg_weighted_val_loss_vectors_sum.shape)
print(avg_main_model_weighted_train_loss_vectors_sum.shape)
print(train_acc_1_sum.shape)
print(val_acc_1_sum.shape)
print(train_acc_2_sum.shape)
print(val_acc_2_sum.shape)
print(prec_train_sum.shape)
print(prec_val_sum.shape)
print("--------- Successfully loaded Sum ----------")




val_loss_bce, train_loss_bce, train_map_bce, val_map_bce, \
   train_loss_vec_bce, val_loss_vec_bce, train_acc_1_bce, val_acc_1_bce, \
    train_acc_2_bce, val_acc_2_bce, prec_train_bce, prec_val_bce = convert_to_df_normal("rerun_logs/bce_acc.txt",True)

# print(train_loss_vec.shape)
# print(val_loss_vec.shape)
# print(train_acc1.shape)
# print(train_acc2.shape)
# print(train_map_.shape)
print(train_loss_vec_bce.shape)
print(val_loss_vec_bce.shape)
print(train_acc_1_bce.shape)
print(val_acc_1_bce.shape)
print(train_acc_2_bce.shape)
print(val_acc_2_bce.shape)
print(prec_train_bce.shape)
print(prec_val_bce.shape)
print("--------- Successfully loaded BCE ----------")


avg_main_model_val_loss_sum, 



if scatplot:
    plot_scatter_plot(prec_val_bce[-1,:], prec_val_max[-1,:], "rerun_logs/plots", "Precision", "BCE", "Max", "max_vs_bce_valprec")
    plot_scatter_plot(prec_train_bce[-1,:], prec_train_max[-1,:], "rerun_logs/plots", "Precision", "BCE", "Max", "max_vs_bce_trainprec")

    plot_scatter_plot(prec_val_bce[-1,:], prec_val_sum[-1,:], "rerun_logs/plots", "Precision", "BCE", "Sum", "sum_vs_bce_valprec")
    plot_scatter_plot(prec_train_bce[-1,:], prec_train_sum[-1,:], "rerun_logs/plots", "Precision", "BCE", "Sum", "sum_vs_bce_trainprec")

    plot_scatter_plot(val_loss_vec_bce[-1,:], avg_unweighted_val_loss_vectors_sum[-1,:],"rerun_logs/plots", "Loss", "BCE", "SUM", "sum_vs_bce_valloss")
    plot_scatter_plot(val_loss_vec_bce[-1,:], avg_unweighted_val_loss_vectors_max[-1,:],"rerun_logs/plots", "Loss", "BCE", "MAX", "max_vs_bce_valloss")


## size if 164 because model_train - train dataset, model_train_ema - train dataset, model_train - val dataset, model_train_ema - val dataset

if(isVal == True and donotShow == False):
    plt.figure(figsize=(10,6))
    plt.plot([p for p in range(len(avg_unweighted_val_loss_max))], avg_unweighted_val_loss_max,label='Unweighted Validation Loss for Max, MAP = 17.22', marker='v')
    #plt.title('Averaged Main Model Unweighted Validation Loss',label='Unweighted Main Model Validation Loss')
    # plt.savefig('unweighted_main_model_validation_loss.png')

    #plt.show()



    plt.plot([p for p in range(len(avg_summed_Up_meta_val_loss_max))], avg_summed_Up_meta_val_loss_max, label='Min Max Objective Value', marker='8')
    #plt.title('Averaged Meta Model Summed up Validation Loss')

    #plt.plot([p for p in range(len(avg_main_model_val_loss_max))], avg_main_model_val_loss_max, label='Main Model Weighted Validation Loss Max Objective', marker='*')
    #plt.plot([p for p in range(len(avg_main_model_weighted_train_loss_max))], avg_main_model_weighted_train_loss_max, label='Main Model Weighted Training Loss Max Objective',marker='D')






    plt.plot([p for p in range(len(avg_unweighted_val_loss_sum))], avg_unweighted_val_loss_sum,label='Unweighted Validation Loss for Sum, MAP = 51.92', marker='X')
    #plt.title('Averaged Main Model Unweighted Validation Loss',label='Unweighted Main Model Validation Loss')
    # plt.savefig('unweighted_main_model_validation_loss.png')

    #plt.show()



    plt.plot([p for p in range(len(avg_summed_Up_meta_val_loss_sum))], avg_summed_Up_meta_val_loss_sum, label='Sum Objective Value', marker='D')
    #plt.title('Averaged Meta Model Summed up Validation Loss')

    #plt.plot([p for p in range(len(avg_main_model_val_loss_sum))], avg_main_model_val_loss_sum, label='Main Model Weighted Validation Loss Sum Objective', marker='*')
    #plt.plot([p for p in range(len(avg_main_model_weighted_train_loss_sum))], avg_main_model_weighted_train_loss_sum, label='Main Model Weighted Training Loss Sum Objective',marker='D')


    plt.plot([p for p in range(len(val_loss_asl))], val_loss_asl, label='ASL Objective Validation Loss, MAP = 66.09', marker='v')
    plt.plot([p for p in range(len(val_loss_bce))], val_loss_bce, label='BCE Objective Validation Loss, MAP = 64.82', marker='o')




    plt.title('Validation Loss Curves Min Max obj, vs sum obj, vs ASL, vs BCE')


    plt.ylabel("Validation Loss magnitude")
    plt.xlabel("Epochs")

    plt.legend(loc='upper right')


    plt.savefig('validation_loss_Curves.png')

    # plt.show()

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


    # plt.savefig('training_loss_Curves.png')

    # plt.show()




if(Showmap):
    plt.figure(figsize=(10,6))
    plt.plot([p for p in range(len(val_map_max))], val_map_max,label=' Mean Avg Precision (Min Max objective)', marker='v')
    #plt.title('Averaged Main Model Unweighted Validation Loss',label='Unweighted Main Model Validation Loss')
    #plt.savefig('unweighted_main_model_validation_loss.png')

    #plt.show()



    plt.plot([p for p in range(len(val_map_sum))], val_map_sum, label='Mean Avg Precision (Sum objective)', marker='8')
    #plt.title('Averaged Meta Model Summed up Validation Loss')

    #plt.plot([p for p in range(len(avg_main_model_val_loss_max))], avg_main_model_val_loss_max, label='Main Model Weighted Validation Loss Max Objective', marker='*')
    #plt.plot([p for p in range(len(avg_main_model_weighted_train_loss_max))], avg_main_model_weighted_train_loss_max, label='Main Model Weighted Training Loss Max Objective',marker='D')








    plt.plot([p for p in range(len(val_map_asl))], val_map_asl, label='ASL Objective mAP', marker='v')
    plt.plot([p for p in range(len(val_map_bce))], val_map_bce, label='BCE Objective mAP', marker='o')




    plt.title('Mean Avg Precision Curves on Val: Min Max obj, vs sum obj, vs ASL, vs BCE')


    plt.ylabel("mAP")
    plt.xlabel("Epochs")

    plt.legend(loc='right')

    plt.savefig('mAP_Curves_val.png')
    plt.close()

    ## plotting for train mAP
    plt.figure(figsize=(10,6))
    plt.plot([p for p in range(len(train_map_max))], train_map_max,label=' Mean Avg Precision (Min Max objective)', marker='v')
    plt.plot([p for p in range (len(train_map_sum))], train_map_sum, label='Mean Avg Precision (Sum objective)', marker='8')
    plt.plot([p for p in range(len(train_map_bce))], train_map_bce, label='BCE Objective mAP', marker='o')

    plt.title('Mean Avg Precision Curves on Train: Min Max obj, vs sum obj, vs ASL, vs BCE')

    plt.ylabel("mAP")
    plt.xlabel("Epochs")
    plt.legend(loc='right')
    plt.savefig('mAP_Curves_train.png')

    # plt.show()
    plt.close()




    # plt.show()
    
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

    # plot_weights(weightarray_sum,os.path.join(savepath, 'weights_sum'))
    # plot_weights(weightsarray_max,os.path.join(savepath, 'weights_max'))

    # weight_class_dict = {}
    #df = pd.DataFrame(weights_max)
    #print(val_loss)
    # df = pd.DataFrame(columns=['idit','class_ids','weights_max','weights_sum'])
    # for iterat in range(len(weights_max)):
        # d = {'idit':iterat,'class_ids': [p for p in range(80)], 'weights_max': weights_max[iterat], 'weights_sum': weights_sum[iterat]}
        #df = pd.DataFrame(data=d)
        # df = df._append(d, ignore_index=True)
        
        
    # print(df[df['idit']==0].head())


    #print(weight_class_dict.values())
    #plt.hist(np.array(list(weight_class_dict.values
    # plt.savefig('weights.png')

    plt.show()
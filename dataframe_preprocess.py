import numpy as np
def convert_to_df(file_name):
    """
    Function to convert the txt file to a dataframe
    :param file_name: name of the txt file
    :return: dataframe
    text file has the following format:
        - Epoch [0/80], Step [000/642], LR 4.0e-06, Loss: 4001.5 (has 6 lines of this format for each epoch and step)
        - mAP score regular 0.0000, map score ema 0.0000
    """
    ## convert txt to csv
    with open(file_name, 'r') as f:
        lines = f.readlines()
    data = []
    unweighted_val_loss = []
    summed_Up_meta_val_loss = []
    weighted_val_loss= []
    main_model_weighted_train_loss =  []
    for i in range(len(lines)):
        if 'mAP' in lines[i]:
            data.append(lines[i].split(' ')[-2:])

        elif 'Main Model Unweighted Validation' in lines[i]:
            #print(lines[i])
            unweighted_val_loss.append(float(lines[i].split(' ')[-1]))
        elif 'Meta Learning Summed up Validation Loss' in lines[i]:
            summed_Up_meta_val_loss.append(float(lines[i].split(' ')[-1]))
        elif 'Main Model Validation Loss' in lines[i]:
            weighted_val_loss.append(float(lines[i].split(' ')[-1]))
        elif 'Main Model Weighted Training Loss' in lines[i]:
            main_model_weighted_train_loss.append(float(lines[i].split(' ')[-1]))
            
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
    return avg_unweighted_val_loss,avg_summed_Up_meta_val_loss,avg_main_model_val_loss,avg_main_model_weighted_train_loss



avg_unweighted_val_loss,avg_summed_Up_meta_val_loss,avg_main_model_val_loss,avg_main_model_weighted_train_loss = (convert_to_df("nohup_outs/objp_l_sum_updated.txt"))

print(len(avg_unweighted_val_loss))
print("Averaged Main Model Unweighted Validation Loss")

import matplotlib.pyplot as plt

plt.plot([p for p in range(len(avg_unweighted_val_loss))], avg_unweighted_val_loss,label='Unweighted Main Model Validation Loss', marker='v')
#plt.title('Averaged Main Model Unweighted Validation Loss',label='Unweighted Main Model Validation Loss')
#plt.savefig('unweighted_main_model_validation_loss.png')

#plt.show()

print("Averaged Meta Model Summed up Validation Loss")


plt.plot([p for p in range(len(avg_summed_Up_meta_val_loss))], avg_summed_Up_meta_val_loss, label='Meta Model Summed up Validation Loss', marker='8')
plt.title('Averaged Meta Model Summed up Validation Loss')

plt.plot([p for p in range(len(avg_main_model_val_loss))], avg_main_model_val_loss, label='Main Model Weighted Validation Loss', marker='*')
plt.plot([p for p in range(len(avg_main_model_weighted_train_loss))], avg_main_model_weighted_train_loss, label='Main Model Weighted Training Loss',marker='D')

plt.title('Loss Curves (Meta/ Vs Main Model)')


plt.legend()
plt.savefig('final_losses_updatedwithmarks.png')

plt.show()
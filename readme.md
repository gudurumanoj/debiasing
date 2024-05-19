### cocostats.py
- Used to get MS COCO dataset stats
- Used to plot anything which concerns with the dataset
- ``` plotfreqBar``` to plot the frequency of the classes in the dataset as a bar graph
-  ``` describemetrics ``` to describe the metrics used to quantify the imbalance of the dataset
- ``` rankcorrelation(cocotrain, cocoval)``` to get the rank correlation between the train and validation set

### helper_functions.py
- Edited ```mAP``` to include classwise confusion metrics

### preprocess_file.py
- Used to preprocess the logs
- ``` convert_to_df_normal ``` for bce and asl
- ``` convert_to_df``` for tried objectives
    - both return all the required metrics, losses, weight vectors (whenever present)


### train.py
Modified the train.py from the original repository to include the following:
- Added the option to train the model with the debiasing loss
- Added arguments to specify the type of objective ``` --type ```
    - asl: normal standard asl objective
    - objp: priliminary objective
    - objpmax: minmax objective
    - bce: binary cross entropy
    - bce2: binary cross entropy with fixed weights learnt from objp
    - objpinv: with fixed weights proportional to the inverse frequency of the classes from the dataset
- Added three functions ```getInverseClassFreqs(), printClassLoss(loss, name), getlearntweights() ``` to get the inverse class frequencies, print the class loss and get the learnt weights to aid in the training process and log the results  
- By default, cofusion stats are logged, use ```--print-info False ``` to change the behaviour
- ``` --seed``` argument to set the seed for reproducibility


# cocostats.py
- Used to get MS COCO dataset stats
- Used to plot anything which concerns with the dataset
- Implemented as ``` coco2014 ``` class which takes in annotation file path as argument.
- ``` plotfreqBar``` to plot the frequency of the classes in the dataset as a bar graph
-  ``` describemetrics ``` to describe the metrics used to quantify the imbalance of the dataset
- ``` print_statistics(self, return_stats=False)``` to print the statistics of the dataset and ``` return_stats``` to return the statistics as a dictionary
- ``` rankcorrelation(cocotrain, cocoval)``` to get the rank correlation between the train and validation set
- Command line arguments to run the file

    - `--annotations_file_path`: Path to the directory containing the COCO 2014 annotations files. Default is `/raid/ganesh/prateekch/MSCOCO_2014`.
    - `--output_dir`: Path to the output directory where plots and statistics will be saved. Default is `/raid/ganesh/prateekch/debiasing/plots/`.
    - `--type`: Type of training objective. Default is `objp`.

# helper_functions.py
- Edited ```mAP``` to include classwise confusion metrics by treating it as a one v all problem
- Used two thresholds to assign labels 0.5 and 0.7 to get the confusion matrix

# preprocess_file.py
- Used to preprocess the logs
- ``` convert_to_df_normal ``` for bce and asl
- ``` convert_to_df``` for tried objectives
    - both return all the required metrics, losses, weight vectors (whenever present) after parsing the log file
- Once the data is obtained, depending on the requirement, the data can be used to plot the graphs
- Usage: ``` python pre

```python
# flags to plot the graphs
isVal           ## to plot the validation data                             
donotShow                         
Showmap         ## to plot the mAP
weightPlot      ## to plot the weight vectors
scatplot        ## to plot the scatter plot
barplot         ## to plot the bar plot
accplot         ## to plot the accuracy plot
plotloss        ## to plot the loss plot
seed            ## to plot for various seeds
```                        


# train.py
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

Detailed description is as follows:

## Arguments
The following arguments can be passed to the script:

- `--data` (default: `'/raid/ganesh/prateekch/MSCOCO_2014'`): Path to the dataset.
- `--lr` (default: `1e-4`): Learning rate.
- `--model-name` (default: `'tresnet_m'`): Model name.
- `--model-path` (default: `'./models_local/MS_COCO_TRresNet_M_224_81.8.pth'`): Path to the pretrained model.
- `--num-classes` (default: `80`): Number of classes in the dataset.
- `-j`, `--workers` (default: `8`): Number of data loading workers.
- `--image-size` (default: `224`): Input image size.
- `--thre` (default: `0.8`): Threshold value.
- `-b`, `--batch-size` (default: `128`): Mini-batch size.
- `--print-freq`, `'-p'` (default: `64`): Print frequency.
- `--type`, `'-t'` (default: `'objp'`): Method type (e.g., 'asl', 'objp', 'bce').
- `--losscrit` (default: `'sum'`): Loss criterion.
- `--reg` (default: `0`): Regularization parameter.
- `--seed` (default: `42`): Random seed.
- `--print-info` (default: `'yes'`): Whether to print information during training.
- `--meta-lr` (default: `1e-4`): Meta-learning rate.

## Functions

### `getInverseClassFreqs()`
Calculates and returns the inverse class frequencies from the COCO 2014 training set.

### `getlearntweights()`
Returns a tensor of pre-learned weights for the classes.

### `main()`
The main function that parses arguments, sets random seeds, initializes models, loads datasets, and starts the training process based on the specified method type.

### `train_multi_label_coco(model, train_loader, val_loader, lr)`
Trains the model using multi-label classification on the COCO dataset with Asymmetric Loss.

#### Parameters:
- `model`: The model to be trained.
- `train_loader`: DataLoader for the training set.
- `val_loader`: DataLoader for the validation set.
- `lr`: Learning rate.

### `validate_multi(val_loader, model, ema_model, print_info="yes", name='')`
Validates the model on the validation set and calculates the mean Average Precision (mAP).

#### Parameters:
- `val_loader`: DataLoader for the validation set.
- `model`: The model to be validated.
- `ema_model`: Exponential Moving Average model.
- `print_info`: Whether to print information during validation.
- `name`: Name for identifying the validation run.

### `printClassLoss(loss, name)`
Prints the class-wise loss for debugging purposes.

#### Parameters:
- `loss`: Loss tensor.
- `name`: Name for identifying the loss printout.

### `class bcelearner(nn.Module)`
A class for training a model using Binary Cross-Entropy Loss.

#### Methods:
- `__init__(self, model_train, train_loader, val_loader, args)`: Initializes the BCE learner.
    - `model_train`: The model to be trained.
    - `train_loader`: DataLoader for the training set.
    - `val_loader`: DataLoader for the validation set.
    - `args`: Parsed arguments.
- `forward(self)`: Trains the model for a specified number of epochs and validates it after each epoch with criterion BCE.

### `class learner(nn.Module)`
A class for training a model using the objective 

#### Methods:
- `__init__(self, model_train, train_loader, val_loader, args)`: Initializes the learner.
    - `model_train`: The model to be trained.
    - `model_val` : The meta model using which the weights are updated
    - `train_loader`: DataLoader for the training set.
    - `val_loader`: DataLoader for the validation set.
    - `args`: Parsed arguments.

- ```forward(self)```: Trains the model for a specified number of epochs and validates it after each epoch with the specified criterion for minmax objective
- ```forwardsum(self)```: Trains the model for a specified number of epochs and validates it after each epoch with the specified criterion for priliminary sum objective and its variants experimented on.
- ```fasttrain``` is a model with same architecture as the ```model_train``` which is used to update parameters of both the models ```model_train and model_val``` using the gradients obtained.

# Logs & Plots
- Logs are saved in the `rerun_logs` directory.
- Plots are saved in the `rerun_logs/plots` directory.



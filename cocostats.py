from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import argparse
from scipy.stats import spearmanr
import seaborn as sns

class coco2014:
    def __init__(self, annotations_file_path):
        self.annotations_file_path = annotations_file_path
        self.coco = COCO(self.annotations_file_path)
        self.categories = self.coco.loadCats(self.coco.getCatIds())
        self.category_ids = [category['id'] for category in self.categories]
        print(self.category_ids)
        self.image_ids = self.coco.getImgIds()
        self.annotations = self.coco.loadAnns(self.coco.getAnnIds())

        self.total_images, self.total_annotations, self.average_annotations_per_image = self.print_statistics(return_stats=True)

        self.cat2cat = dict()                       ## to map category ids to integer indices, only for internal usage
        for cat in self.coco.cats.keys():
            self.cat2cat[cat] = len(self.cat2cat)


        self.class_frequencies = {self.cat2cat[category_id]: 0 for category_id in self.category_ids}
        self.get_class_frequencies()

    def get_class_frequencies(self):
        for img_id in self.image_ids:
            s = set()
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)
            for ann in anns:
                s.add(self.cat2cat[ann['category_id']])
            for i in s:
                self.class_frequencies[i]+=1

    def print_statistics(self, return_stats=False):
        total_images = len(self.image_ids)
        total_annotations = len(self.annotations)
        average_annotations_per_image = total_annotations / total_images

        

        if not return_stats:
            print("Total images:", total_images)
            print("Total annotations:", total_annotations)
            print("Average annotations per image:", average_annotations_per_image)
        else:
            return total_images, total_annotations, average_annotations_per_image


    def print_class_frequencies(self):
        for category_id, frequency in self.class_frequencies.items():
            print(category_id, frequency)

        avg = sum(self.class_frequencies.values())/len(self.class_frequencies)

        print("Average class frequency:", avg)

def getweight():
    a = np.array([0.3121, 0.3381, 0.3539, 0.3975, 0.3942, 0.3723, 0.3953, 0.3936, 0.3952,
        0.3797, 0.3778, 0.3904, 0.3856, 0.3850, 0.3833, 0.3909, 0.3529, 0.3916,
        0.3914, 0.3808, 0.3875, 0.3818, 0.3902, 0.3916, 0.3744, 0.3807, 0.3647,
        0.3902, 0.3921, 0.3592, 0.3655, 0.3797, 0.3817, 0.3965, 0.3732, 0.3832,
        0.3616, 0.3940, 0.3767, 0.3563, 0.3907, 0.3829, 0.3889, 0.3850, 0.3939,
        0.4001, 0.3997, 0.3847, 0.3876, 0.3851, 0.3848, 0.3791, 0.3945, 0.3903,
        0.3987, 0.3666, 0.3896, 0.3845, 0.3314, 0.3927, 0.4033, 0.3494, 0.3899,
        0.3767, 0.3981, 0.3809, 0.3804, 0.4099, 0.3901, 0.3835, 0.3959, 0.3839,
        0.3967, 0.3696, 0.4109, 0.4019, 0.3910, 0.3714, 0.3929, 0.3849])

    return a

def plot(weightfunc,outputdir, type):   ## weightfunc is the weight function to be used
    # weight = getweight()
    weight = weightfunc()

    trainpath = os.path.join(outputdir,type,'train')
    valpath = os.path.join(outputdir,type,'val')

    # train class frequencies
    x = np.array([i for i in range(80)])

    indexes = np.array(list(coco2014_train.class_frequencies.values())).argsort()


    ## without sorting
    fig, axs = plt.subplots(2, 1, figsize=(20, 20))
    fig.suptitle('Class Frequencies and Weights')
    axs[0].bar(x,np.array(list(coco2014_train.class_frequencies.values())))
    axs[0].set_title('Train Class Frequencies')
    axs[0].set_xlabel('Class Index')
    axs[0].set_ylabel('Frequency')
    axs[1].bar(x,weight)
    axs[1].set_title('Train Class Weights')
    axs[1].set_xlabel('Class Index')
    axs[1].set_ylabel('Weight')

    # plt.savefig('class_freq_weight_train_sort.png')
    plt.savefig(os.path.join(trainpath,'class_freq_weight_train.png'))

    fig, axs = plt.subplots(2, 1, figsize=(20, 20))
    fig.suptitle('Class Frequencies and Weights')
    axs[0].bar(x,np.array(list(coco2014_train.class_frequencies.values()))[indexes])
    axs[0].set_title('Train Class Frequencies')
    axs[0].set_xlabel('Class Index')
    axs[0].set_ylabel('Frequency')
    axs[1].bar(x,weight[indexes])
    axs[1].set_title('Train Class Weights')
    axs[1].set_xlabel('Class Index')
    axs[1].set_ylabel('Weight')

    # plt.savefig('class_freq_weight_train_sort.png')
    plt.savefig(os.path.join(trainpath,'class_freq_weight_train_sort.png'))

    ## Manual weights assigned according to inverse frequencies
    invFreqWeights = 1/np.array(list(coco2014_train.class_frequencies.values()))

    fig, axs = plt.subplots(2, 1, figsize=(20, 20))
    fig.suptitle('Class Frequencies and Weights')
    axs[0].bar(x,invFreqWeights[indexes])
    axs[0].set_title('Train Class Inverse Frequency Weights')
    axs[0].set_xlabel('Class Index')
    axs[0].set_ylabel('Frequency')
    axs[1].bar(x,weight[indexes])
    axs[1].set_title('Train Class Weights')
    axs[1].set_xlabel('Class Index')
    axs[1].set_ylabel('Weight')

    plt.savefig(os.path.join(trainpath,'class_inv_freq_weight_train_sort_sum.png'))

    ## val class frequencies
    indexes = np.array(list(coco2014_val.class_frequencies.values())).argsort() 
    fig, axs = plt.subplots(2, 1, figsize=(20, 20))

    fig.suptitle('Class Frequencies and Weights')
    # axs[0].bar(coco2014_val.class_frequencies.keys(), coco2014_val.class_frequencies.values())
    axs[0].bar(x,np.array(list(coco2014_val.class_frequencies.values()))[indexes])
    axs[0].set_title('Val Class Frequencies')
    axs[0].set_xlabel('Class Index')
    axs[0].set_ylabel('Frequency')
    # axs[1].bar(coco2014_val.class_frequencies.keys(), weight)
    axs[1].bar(x,weight[indexes])
    axs[1].set_title('Val Class Weights')
    axs[1].set_xlabel('Class Index')
    axs[1].set_ylabel('Weight')

    # plt.savefig('class_freq_weight_val_sort.png')
    plt.savefig(os.path.join(valpath,'class_freq_weight_val_sort.png'))


    ## plotting with the ratio of positive to negative imbalance
    a = len(coco2014_train.image_ids) - np.array(list(coco2014_train.class_frequencies.values()))
    b = np.array(list(coco2014_train.class_frequencies.values()))
    print(a)
    print(b)

    arr = b/a
    print(arr)
    inde = arr.argsort()
    fig, axs = plt.subplots(2, 1, figsize=(20, 20))

    print(inde)
    print(arr[inde])

    fig.suptitle('Class Frequencies and Weights')
    # axs[0].bar(coco2014_train.class_frequencies.keys(), b/a)
    axs[0].bar(x,b[indexes]/a[indexes])
    axs[0].set_title('Train Class Imbalance ratio')
    axs[0].set_xlabel('Class Index')
    axs[0].set_ylabel('Frequency')
    # axs[1].bar(coco2014_train.class_frequencies.keys(), weight)
    axs[1].bar(x,weight[indexes])
    axs[1].set_title('Train Class Weights')
    axs[1].set_xlabel('Class Index')
    axs[1].set_ylabel('Weight')

    # plt.savefig('class_freq_weight_train_ratio_sort.png')
    plt.savefig(os.path.join(trainpath,'class_freq_weight_train_ratio_sort.png'))

    ## val class frequencies
    a = len(coco2014_val.image_ids) - np.array(list(coco2014_val.class_frequencies.values()))
    b = np.array(list(coco2014_val.class_frequencies
                .values()))

    indexes = (b/a).argsort()

    fig, axs = plt.subplots(2, 1, figsize=(20, 20))
    fig.suptitle('Class Frequencies and Weights')
    # axs[0].bar(coco2014_val.class_frequencies
                # .keys(), b/a)
    axs[0].bar(x,b[indexes]/a[indexes])
    axs[0].set_title('Val Class Imbalance ratio')
    axs[0].set_xlabel('Class Index')
    axs[0].set_ylabel('Frequency')
    # axs[1].bar(coco2014_val.class_frequencies.keys(), weight)
    axs[1].bar(x,weight[indexes])
    axs[1].set_title('Val Class Weights')
    axs[1].set_xlabel('Class Index')
    axs[1].set_ylabel('Weight')


    # plt.savefig('class_freq_weight_val_ratio_sort.png')
    plt.savefig(os.path.join(valpath,'class_freq_weight_val_ratio_sort.png'))

def rankcorrelation(cocotrain, cocoval):
    training_freq = list(cocotrain.values())
    validation_freq = list(cocoval.values())

    # Rank the frequencies
    training_rank = [sorted(training_freq).index(x) + 1 for x in training_freq]
    validation_rank = [sorted(validation_freq).index(x) + 1 for x in validation_freq]

    # print(training_rank)
    # print(validation_rank)

    # Compute the rank correlation coefficient
    correlation_coefficient, p_value = spearmanr(training_rank, validation_rank)

    return correlation_coefficient, p_value


def plotfreqBar(cocotrain, cocoval, savepath='/raid/ganesh/prateekch/debiasing/'):
    ## plot a bar plot of the frequencies of the classes with train and val side by side for each class
    classes = list(cocotrain.class_frequencies.keys())
    training_freq = np.array(list(cocotrain.class_frequencies.values()))
    validation_freq = np.array(list(cocoval.class_frequencies.values()))

    sns.set_theme()
    # Set the width of the bars
    bar_width = 0.3

    # Set the positions of the bars on the x-axis
    r1 = np.arange(len(classes))
    # r2 = [x + bar_width for x in r1]


    # Create the bar plot
    plt.figure(figsize=(15,9))
    plt.bar(r1, training_freq, color='b', width=bar_width, label='Train')
    plt.bar(r1+bar_width, validation_freq, color='r', width=bar_width, label='Validation')

    # Add labels and title
    plt.xlabel('Classes')
    plt.ylabel('Frequencies')
    plt.xticks([r + bar_width/2 for r in range(len(classes))], classes)
    plt.title('Train vs Validation Frequencies for Each Class')

    plt.legend()

    # Save the plot
    plt.savefig(os.path.join(savepath, 'train_val_freq.png'))

    ## plotting it by normalizing the frequencies
    plt.close()

    plt.figure(figsize=(15,9))
    plt.bar(r1, training_freq/cocotrain.total_images, color='b', width=bar_width, label='Train',align='center')
    plt.bar(r1+bar_width, validation_freq/cocoval.total_images, color='r', width=bar_width, label='Validation',align='center')

    # Add labels and title
    plt.xlabel('Classes' )
    plt.ylabel('Frequencies' )
    plt.xticks([r + 0.1 for r in range(len(classes))], classes)
    plt.title('Train vs Validation fraction of positive samples for Each Class')

    plt.legend()

    # Save the plot
    plt.savefig(os.path.join(savepath, 'train_val_normalised.png'))

    plt.close()
    ## same above but without class 0
    classes = classes[1:]
    training_freq = training_freq[1:]
    validation_freq = validation_freq[1:]

    plt.figure(figsize=(15,9))
    plt.bar(r1[1:], training_freq/cocotrain.total_images, color='b', width=bar_width, label='Train',align='center')
    plt.bar(r1[1:]+bar_width, validation_freq/cocoval.total_images, color='r', width=bar_width, label='Validation',align='center')
    plt.xlabel('Classes')
    plt.ylabel('Frequencies')
    plt.xticks([r + 0.1 for r in range(len(classes))], classes)
    plt.title('Train vs Validation fraction of positive samples for Each Class (excluding class 0)')
    plt.legend()
    plt.savefig(os.path.join(savepath, 'train_val_normalised_no0.png'))


def describemetrics(class_frequencies, name, tot=82783):
    ## class_frequencies is a dict with class index as key and frequency as value
    print("---------------", name, "Statistics---------------")

    print("Mean:", np.mean(list(class_frequencies.values())))

    if name == "Val":
        tot = 40504
    ## intra imbalance, for each class, the ratio of max(positive,negative) to min(positive, negative) samples
    freq = np.array(list(class_frequencies.values()))

    print("Max Frequency:", np.max(freq))
    print("Min Frequency:", np.min(freq))

    a = tot - freq

    intraimb = np.array([np.max([freq[i],a[i]])/np.min([freq[i],a[i]]) for i in range(len(freq))])

    print("Intra Imbalance:", np.mean(intraimb))
    print("Intra Imbalance max:", np.max(intraimb))

    ## inter imbalance, for each class, the ratio of max(freq) to freq[i] for all i
    interimb = np.array([np.max(freq)/freq[i] for i in range(len(freq))])

    print("Inter Imbalance:", np.mean(interimb))

    # print("Intra Imbalance:", np.max(a,freq)/np.min(a,freq))





if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser(description='COCO 2014 statistics')
    parser.add_argument('--annotations_file_path', type=str, default='/raid/ganesh/prateekch/MSCOCO_2014',
                        help='path to the annotations file for the MSCOCO 2014 dataset')
    parser.add_argument('--output_dir', type=str, default='/raid/ganesh/prateekch/debiasing/plots/',
                        help='path to the output directory where the statistics will be saved')
    parser.add_argument('--type', type=str, default='objp/distribution', help='type of training objective')


    args = parser.parse_args()
    
    train_annotation_file_path = os.path.join(args.annotations_file_path, 'annotations/instances_train2014.json')
    val_annotation_file_path = os.path.join(args.annotations_file_path, 'annotations/instances_val2014.json')


    coco2014_train = coco2014(train_annotation_file_path)
    coco2014_val = coco2014(val_annotation_file_path)

    print("---------------Train Statistics---------------")
    coco2014_train.print_statistics()
    coco2014_train.print_class_frequencies()

    print("---------------Val Statistics---------------")
    coco2014_val.print_statistics()
    coco2014_val.print_class_frequencies()

    describemetrics(coco2014_train.class_frequencies, "Train")
    describemetrics(coco2014_val.class_frequencies, "Val")

    rankcorrelation = 0
    plotfreq = 0

    if rankcorrelation:


        print("---------------Rank Correlation---------------")
        correlation_coefficient, p_value = rankcorrelation(coco2014_train.class_frequencies, coco2014_val.class_frequencies)
        print("Rank correlation coefficient:", correlation_coefficient)
        print("P-value:", p_value)

        """

            Rank correlation coefficient: 0.990693749408758
            P-value: 2.522595515473387e-69

        """

    if plotfreq:

        # print("---------------Plotting Frequencies---------------")
        plot(getweight, args.output_dir, args.type)


from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import argparse

# Parse arguments
parser = argparse.ArgumentParser(description='COCO 2014 statistics')
parser.add_argument('--annotations_file_path', type=str, default='/raid/ganesh/prateekch/debiasing/MSCOCO_2014',
                    help='path to the annotations file for the MSCOCO 2014 dataset')
parser.add_argument('--output_dir', type=str, default='/raid/ganesh/prateekch/debiasing/plots/',
                    help='path to the output directory where the statistics will be saved')
parser.add_argument('--type', type=str, default='objp', help='type of training objective')


args = parser.parse_args()

class coco2014:
    def __init__(self, annotations_file_path):
        self.annotations_file_path = annotations_file_path
        self.coco = COCO(self.annotations_file_path)
        self.categories = self.coco.loadCats(self.coco.getCatIds())
        self.category_ids = [category['id'] for category in self.categories]
        print(self.category_ids)
        self.image_ids = self.coco.getImgIds()
        self.annotations = self.coco.loadAnns(self.coco.getAnnIds())

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

    def print_statistics(self):
        total_images = len(self.image_ids)
        total_annotations = len(self.annotations)
        average_annotations_per_image = total_annotations / total_images
        print("Total images:", total_images)
        print("Total annotations:", total_annotations)

        # print(self.cat2cat)
        # self.print_class_frequencies()

    # def print_class_frequencies_indexed(self):
    #     for category_id, frequency in self.class_freq_indexed.items():
    #         print(category_id, frequency)

    def print_class_frequencies(self):
        for category_id, frequency in self.class_frequencies.items():
            print(category_id, frequency)

def getweight():
    a = np.array([1.7315e-02, 7.7686e-04, 6.1454e-03, 5.1478e-04, 2.6118e-04, 7.0636e-04,
        1.8101e-04, 5.2400e-04, 6.2274e-04, 1.7365e-04, 1.9063e-04, 6.3221e-04,
        9.6055e-05, 9.8196e-04, 1.2020e-03, 2.2958e-03, 2.1715e-03, 4.0094e-03,
        1.0940e-03, 8.7631e-04, 2.2662e-04, 2.4457e-03, 3.6053e-04, 1.0707e-03,
        7.0092e-04, 3.3471e-04, 1.5146e-03, 1.6393e-04, 2.2784e-04, 3.8108e-04,
        1.6104e-03, 2.9130e-04, 1.6954e-03, 1.0924e-03, 1.0467e-03, 7.6482e-03,
        1.4336e-04, 9.0938e-04, 9.8308e-04, 1.7667e-02, 3.8589e-01, 3.7264e-01,
        9.9055e-01, 8.8490e-01, 9.9899e-01, 9.8760e-01, 3.9794e-04, 4.5660e-04,
        5.9959e-04, 4.9784e-04, 2.0928e-04, 8.4627e-04, 5.0633e-04, 1.1565e-03,
        2.0378e-04, 1.3017e-03, 5.9122e-02, 5.0844e-03, 6.7782e-02, 2.6906e-03,
        8.3072e-01, 4.3835e-04, 2.0616e-02, 5.0451e-03, 1.4188e-02, 1.6356e-03,
        1.0313e-02, 9.0151e-03, 9.8474e-01, 9.9748e-01, 1.2497e-04, 5.1923e-04,
        6.9381e-01, 1.4629e-01, 9.9711e-01, 9.9999e-01, 1.0000e+00, 9.9998e-01,
        2.3133e-03, 1.2250e-02])

    return a

def plot(weightfunc,outputdir, type):   ## weightfunc is the weight function to be used
    # weight = getweight()
    weight = weightfunc()

    trainpath = os.path.join(outputdir,type,'train')
    valpath = os.path.join(outputdir,type,'val')

    # train class frequencies
    x = np.array([i for i in range(80)])

    indexes = np.array(list(coco2014_train.class_frequencies.values())).argsort()

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

    fig, axs = plt.subplots(2, 1, figsize=(20, 20))
    indexes = (b/a).argsort()

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


train_annotation_file_path = os.path.join(args.annotations_file_path, 'annotations/instances_train2014.json')
val_annotation_file_path = os.path.join(args.annotations_file_path, 'annotations/instances_val2014.json')


coco2014_train = coco2014(train_annotation_file_path)
coco2014_val = coco2014(val_annotation_file_path)

print("---------------Train Statistics---------------")
coco2014_train.print_statistics()

print("---------------Val Statistics---------------")
coco2014_val.print_statistics()

plot(args.output_dir, args.type)


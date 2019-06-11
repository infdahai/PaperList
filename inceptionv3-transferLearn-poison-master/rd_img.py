import os

from scipy.misc import imsave
import numpy as np
import  cv2

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dic = pickle.load(fo, encoding='bytes')
    return dic


dogDir = './Data/rawImages/main_dog/'
fishDir = './Data/rawImages/fish/'

file_path = 'cifar-10-batches-py/'
file_name_test = 'test_batch'
file_name_label = 'batches.meta'
file_name_data = 'data_batch_%d'

label_name_dic: dict = unpickle(file_path + file_name_label)
new_list: list = label_name_dic[b'label_names']
label_index = [new_list.index(b'dog'), new_list.index(b'cat')]

if not os.path.exists(dogDir):
    os.makedirs(dogDir)
if not os.path.exists(fishDir):
    os.makedirs(fishDir)

num = 10000
dim = 3072
label_dic = unpickle(file=file_path + file_name_test)
for i in range(1, 6):
    data_dic: dict = unpickle(file_path + file_name_data % i)
    fileNames: list = data_dic[b'filenames']
    labels: list = data_dic[b'labels']
    data: np.ndarray = data_dic[b'data']
    dog_list: list = [i for i in range(num) if labels[i] == label_index[0]]
    cat_list: list = [j for j in range(num) if labels[j] == label_index[1]]
    for i in dog_list:
        dog_name: str = fileNames[i].decode('utf-8')
        img_arr = data[i].reshape(32, 32, 3)
        cv2.imwrite(dogDir+dog_name, img_arr)

    for j in cat_list:
        cat_name: str = fileNames[j].decode('utf-8')
        img_arr = data[j].reshape(32, 32, 3)
        cv2.imwrite(fishDir+cat_name, img_arr)
print()

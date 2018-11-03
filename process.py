import os
import numpy as np
import csv
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import pickle
from PIL import Image, ImageDraw

def trans2val(path):
    data_dir = path
    file_list = os.listdir(data_dir)
    for file in file_list:
        coor_list = []
        class_list = []
        if file.split('.')[-1] == 'csv':
            try:
                content = pd.read_csv(data_dir + file, engine='python')
                coors = content['region_shape_attributes']
                classes = content['region_attributes']
                for coor, class_ in zip(coors, classes):
                    c = eval(coor)
                    if c['name'] != 'polygon':
                        break
                    coor_list.append(c)
                    class_ = eval(class_)['class']
                    if class_ == '' or class_ is None:
                        break
                    class_list.append(class_)

                if len(class_list) == 0:
                    continue
                with open(data_dir + file.split('.')[0] + '.val', 'wb') as f:
                    pickle.dump((coor_list, class_list), f)
            except BaseException:
                # print("continue")
                pass


def get_mask(path):
    data_dir = path
    file_list = os.listdir(data_dir)
    for file in file_list:
        if file.split('.')[-1] == 'val':
            coors, classes = pickle.load(open(data_dir + file, 'rb'))
            print(file)
            print(coors)
            print(classes)
            image = Image.open(data_dir + file.split('.')[0] + '.jpg')
            draw = ImageDraw.Draw(image)
            for c in coors:
                coor_list = []
                for x, y in zip(c['all_points_x'], c['all_points_y']):
                    coor_list.append(x)
                    coor_list.append(y)
                draw.polygon(coor_list)
            plt.figure(figsize=(12, 12))
            plt.imshow(image)
            plt.show()

# 这行代码只用运行一次，并且需要先把所有的csv文件放到和图片一个目录下，会生成对应的.val文件
# trans2val('F:/slience/天池标注/')
# 显示并获取框框和类别
get_mask('F:/slience/天池标注/')
import pandas as pd
import glob
import os
import numpy as np
import json
mapper = {} # use maper to store cat
def csvParse(csvPath):
    df = pd.read_csv(csvPath, engine='python', dtype={'region_shape_attributes': dict, 'region_attributes': dict})
    bboxes = np.empty([df.shape[0], 4])
    category = np.empty([df.shape[0], 1])
    for i, (box, className) in enumerate(zip(df['region_shape_attributes'], df['region_attributes'])):
        className = eval(className)
        box = eval(box)
        _class = className.get("") if "" in className.keys() else className.get('class')
        if _class is None:
            print(csvPath)
            continue
        if not mapper.get(_class):
            mapper[_class] = len(mapper.keys())+1 # label of BG is 0
        category[i] = mapper[_class]
        if box['name']!='polygon':
            continue
        xarray = np.array(box['all_points_x'])
        yarray = np.array(box['all_points_y'])
        x1, x2 = np.min(xarray), np.max(xarray)
        y1, y2 = np.min(yarray), np.max(yarray)

        bboxes[i, ] = np.array([x1, y1, x2, y2])
    return bboxes, category

def parseDataset(csvroot, imgroot):
    """
    :param csvroot:
    :param imgroot:
    :return:
    data structure
    file_name: str, full path to the image
    boxes: numpy array of kx4 floats
    class: numpy array of k integers
    """
    csvFiles = glob.glob(os.path.join(csvroot, '*.csv'))
    data = []
    for file in csvFiles:
        fileName = os.path.split(file)[-1].split('.')[0]
        imgPath = os.path.join(imgroot, fileName+'.jpg')
        boxes, category = csvParse(file)
        data.append({'file_name': imgPath, 'boxes': boxes, 'class': category})
    return data

if __name__=='__main__':
    data = \
        parseDataset('F:\slience\天池标注',
                     'F:\Chrome Download\guangdong_round1_train1_20180903\guangdong_round1_train1_20180903')
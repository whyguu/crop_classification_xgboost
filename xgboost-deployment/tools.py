# coding: utf-8
import os
import numpy as np
import tifffile
from skimage import io
import argparse
from color_bar import *

try:
    import gdal
except:
    pass

try:
    import matplotlib.pyplot as plt
    import itertools
except:
    pass


# argument
def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-image', type=str, required=False, default='./org1.tif')  # img.tif
    parser.add_argument('-label', type=str, required=False, default='./lb1.tif')  # ./img_classes.tif
    parser.add_argument('-output', type=str, required=False, default='./haha_out.tif')
    return parser.parse_args()


# read
def read_tiff(path):
    return np.squeeze(io.imread(path, ))


# gdal image
# 读图像文件
def read_img(filename, only_coordinate=False):
    dataset = gdal.Open(filename)  # 打开文件

    im_width = dataset.RasterXSize  # 栅格矩阵的列数
    im_height = dataset.RasterYSize  # 栅格矩阵的行数

    im_geotrans = dataset.GetGeoTransform()  # 仿射矩阵
    im_proj = dataset.GetProjection()  # 地图投影信息
    if only_coordinate:
        del dataset
        return im_proj, im_geotrans

    im_data = dataset.ReadAsArray(0, 0, im_width, im_height)  # 将数据写成数组，对应栅格矩阵
    if len(im_data.shape) == 3:
        im_data = im_data.transpose(1, 2, 0)

    del dataset
    return im_proj, im_geotrans, im_data


# 写文件，以写成tif为例
def write_img(filename, im_proj, im_geotrans, im_data):
    # gdal数据类型包括
    # gdal.GDT_Byte,
    # gdal .GDT_UInt16, gdal.GDT_Int16, gdal.GDT_UInt32, gdal.GDT_Int32,
    # gdal.GDT_Float32, gdal.GDT_Float64

    # 判断栅格数据的数据类型
    if 'uint8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
        # print('uint8')
    elif 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    # 判读数组维数
    if len(im_data.shape) == 3:
        im_data = im_data.transpose(2, 0, 1)
        im_bands, im_height, im_width = im_data.shape
    else:
        im_bands, (im_height, im_width) = 1, im_data.shape

    # 创建文件
    driver = gdal.GetDriverByName("GTiff")  # 数据类型必须有，因为要计算需要多大内存空间
    dataset = driver.Create(
        filename,
        im_width,
        im_height,
        im_bands,
        datatype)

    dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
    dataset.SetProjection(im_proj)  # 写入投影

    if im_bands == 1:
        dataset.GetRasterBand(1).WriteArray(im_data)  # 写入数组数据
    else:
        for i in range(im_bands):
            dataset.GetRasterBand(i + 1).WriteArray(im_data[i])

    del dataset


# plot confusion_matrix
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if cmap is None:
        cmap = plt.cm.Blues
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('cm.png')


def pseudo_map(pred, class_idx, path):
    colors = np.array(COLOR_BAR2).reshape(-1, 3)
    # print(colors)
    colors = colors[0:len(class_idx)]
    pseudo = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
    # print(colors)
    for i, idx in enumerate(class_idx):
        pseudo[pred == idx, :] = colors[i, :]

    io.imsave(path, pseudo)


if __name__ == "__main__":
    # cm = np.load('cm.npy')
    # plot_confusion_matrix(cm, classes=['1', '2', '3', '4', '5', '6', '7', '9', '10', '11'], normalize=True)
    # exit(0)
    mp = np.squeeze(io.imread('/Users/whyguu/Desktop/haha_out.tif'))
    mp1 = np.squeeze(io.imread('/Users/whyguu/Desktop/lb1.tif'))
    mp = np.concatenate((mp, mp1), axis=1)
    pseudo_map(mp, [1, 2, 3, 4, 5, 6, 7, 9, 10, 11], './pseudo.png')
    exit(0)
    proj, geotrans, _ = read_img('../data/haidian_data/data/2003.73.tif')  # 读数据
    data2 = tifffile.imread('../data/haidian_data/label/2003.73.tif')
    data2.astype('uint8')
    print(data2.shape)
    write_img('abc.tif', proj, geotrans, data2)  # 写数据
    data = tifffile.imread('abc.tif')
    print(data.shape)

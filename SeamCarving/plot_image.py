import json
import os
from os import path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties

font = FontProperties(fname=r"SimHei.ttf", size=14)

dic = {}

try:
    with open(path.join('images', 'output', 'file.json'), 'r') as f:
        dic = json.loads(f.read())
except:
    pass


def cv_save_image(image, folder, file_name):
    prefix = path.join('images', 'output')
    folder = path.join(prefix, folder) if folder is not None else prefix
    if not path.exists(folder):
        os.makedirs(folder)
    if folder not in dic.keys():
        dic[folder] = []
    dic[folder].append(path.join(folder, file_name))
    cv2.imwrite(path.join(folder, file_name), cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2BGR))


def cv_get_files(folder):
    """
    获取文件夹下所有添加过的文件
    :param folder: 文件夹
    :return: 文件列表
    """
    return dic[folder]


def cv_save_files():
    with open(path.join('images', 'output', 'file.json'), 'w') as f:
        f.write(json.dumps(dic))


def plot_image(image, gray=False, title=None, folder=None, file_name=None):
    """
    绘制图片
    :param image: 图片数据
    :param gray: 是否为灰度图
    :param title: 图片标题
    :param folder: 图片保存文件夹
    :param file_name: 保存文件名称
    :return:
    """
    plt.figure(figsize=(12, 12))
    if gray:
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(image)
    if title is not None:
        plt.title(title, fontproperties=font)
    if file_name is not None:
        prefix = path.join('images', 'output')
        folder = path.join(prefix, folder) if folder is not None else prefix
        if not path.exists(folder):
            os.makedirs(folder)
        plt.savefig(path.join(folder, file_name))
        pass
    # plt.show()
    plt.close()

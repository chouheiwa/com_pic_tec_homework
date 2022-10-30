from os import path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties

font = FontProperties(fname=r"SimHei.ttf", size=14)


def cv_save_image(image, file_name):
    cv2.imwrite(file_name, cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2BGR))


def plot_image(image, gray=False, title=None, file_name=None):
    """
    绘制图片
    :param image: 图片数据
    :param gray: 是否为灰度图
    :param title: 图片标题
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
        plt.savefig(path.join('images', 'output', file_name))
        pass
    # plt.show()
    plt.close()

from datetime import datetime
from os import path

import cv2
import numpy as np
import plot_image as pi


def measure_times(func, func_name, *args, **kwargs):
    """
    测量函数执行时间
    :param func: 函数
    :param func_name: 函数名称
    :param args: 参数
    :param kwargs: 参数
    :return: 函数执行时间
    """
    start = datetime.now()
    result = func(*args, **kwargs)
    end = datetime.now()
    print(f'{func_name} cost time: {end - start}')
    return result


def mark_pixel(image, seam):
    """
    标记像素
    :param image: 图片源数据
    :param seam: 缝数据
    :return: 标记后的图片数据
    """
    image_copy = image.copy()
    for i in range(seam.shape[0]):
        image_copy[seam[i, 0], seam[i, 1]] = [255, 0, 0]
    return image_copy


# noinspection PyMethodMayBeStatic
class SeamCarvingBase:
    """基类，子类需要实现"""
    name = 'base'

    def __init__(self, file_path):
        self.file_path = file_path
        self.image = self.read_image()

    def read_image(self):
        """
        读取图片
        :param file: 图片文件
        :return: 图片数据
        """
        img = cv2.imread(self.file_path)
        # 这里由于opencv 读取图片后的通道是BGR，而matplotlib显示图片的通道是RGB，所以需要转换一下
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def convert_to_grayscale(self, image):
        """
        将图片转换为灰度图
        :param image: 图片源数据
        :return: 灰度图数据
        """
        # 这里使用opencv的cvtColor函数将图片转换为灰度图
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # 此处代码为自己实现的灰度图转换方法，效率较低
        # gray_image = np.zeros((image.shape[0], image.shape[1]), dtype=np.dtype('i4'))
        # for i in range(image.shape[0]):
        #     for j in range(image.shape[1]):
        #         gray_value = 0.299 * image[i, j, 0] + 0.587 * image[i, j, 1] + 0.114 * image[i, j, 2]
        #         gray_image[i, j] = math.floor(gray_value)
        #
        # return gray_image

    def calculate_energy(self, gray_image) -> np.ndarray:
        """
        计算能量方法，此方法为抽象方法，计算因为计算能量的方法不同，所以需要子类实现
        :param gray_image: 灰度图数据
        :return: 能量数据
        """
        assert '需要被子类实现'
        pass

    def calculate_cumulative_energy(self, energy_image, axis=1):
        """
        计算累计能量
        :param axis: 1为竖直方向 0为水平方向
        :param energy_image: 能量数据
        :return: 累计能量数据
        """
        cumulative_energy = np.zeros((energy_image.shape[0], energy_image.shape[1]))
        if axis == 1:
            cumulative_energy[0, :] = energy_image[0, :]
            for i in range(1, energy_image.shape[0]):
                for j in range(energy_image.shape[1]):
                    if j == 0:
                        cumulative_energy[i, j] = energy_image[i, j] + min(cumulative_energy[i - 1, j],
                                                                           cumulative_energy[i - 1, j + 1])
                    elif j == energy_image.shape[1] - 1:
                        cumulative_energy[i, j] = energy_image[i, j] + min(cumulative_energy[i - 1, j],
                                                                           cumulative_energy[i - 1, j - 1])
                    else:
                        cumulative_energy[i, j] = energy_image[i, j] + min(cumulative_energy[i - 1, j],
                                                                           cumulative_energy[i - 1, j - 1],
                                                                           cumulative_energy[i - 1, j + 1])
        else:
            cumulative_energy[:, 0] = energy_image[:, 0]
            for j in range(1, energy_image.shape[1]):
                for i in range(energy_image.shape[0]):
                    if i == 0:
                        cumulative_energy[i, j] = energy_image[i, j] + min(cumulative_energy[i, j - 1],
                                                                           cumulative_energy[i + 1, j - 1])
                    elif i == energy_image.shape[0] - 1:
                        cumulative_energy[i, j] = energy_image[i, j] + min(cumulative_energy[i, j - 1],
                                                                           cumulative_energy[i - 1, j - 1])
                    else:
                        cumulative_energy[i, j] = energy_image[i, j] + min(cumulative_energy[i - 1, j - 1],
                                                                           cumulative_energy[i, j - 1],
                                                                           cumulative_energy[i + 1, j - 1])

        return cumulative_energy

    def calculate_seam(self, cumulative_energy, axis=1):
        """
        通过累计能量计算能量最低的路径
        :param cumulative_energy: 累计能量图
        :param axis: 方向
        :return:
        """
        seam = np.zeros((cumulative_energy.shape[0],), dtype=np.dtype('i4'))
        last_index = cumulative_energy.shape[0 if axis == 1 else 1] - 1
        indexes = np.arange(last_index + 1)
        for i in range(last_index, -1, -1):
            if i == last_index:
                data = cumulative_energy[i, :] if axis == 1 else cumulative_energy[:, i]
                seam[i] = np.argmin(data)
            else:
                if seam[i + 1] == 0:
                    data = cumulative_energy[i, seam[i + 1]:seam[i + 1] + 2]
                    if axis == 0:
                        data = cumulative_energy[seam[i + 1]:seam[i + 1] + 2, i]
                    seam[i] = seam[i + 1] + np.argmin(data)
                elif seam[i + 1] == cumulative_energy.shape[1] - 1:
                    data = cumulative_energy[i, seam[i + 1] - 1:seam[i + 1] + 1]
                    if axis == 0:
                        data = cumulative_energy[seam[i + 1] - 1:seam[i + 1] + 1, i]
                    seam[i] = np.argmin(data) + seam[i + 1] - 1
                else:
                    data = cumulative_energy[i, seam[i + 1] - 1:seam[i + 1] + 2]
                    if axis == 0:
                        data = cumulative_energy[seam[i + 1] - 1:seam[i + 1] + 2, i]
                    seam[i] = np.argmin(data) + seam[i + 1] - 1
        return np.column_stack((indexes, seam)) if axis == 1 else np.column_stack((seam, indexes))

    def remove_seam(self, image, seam):
        """
        移除缝
        :param image: 图片源数据
        :param seam: 缝数据
        :return: 移除缝后的图片数据
        """
        image_list = image.tolist()
        for item in seam:
            del image_list[item[0]][item[1]]
        return np.array(image_list)

    def seam_carving(self, image, new_width, new_height, show_step=False):
        """
        图像缩放
        :param image: 图片源数据
        :param new_width: 新的宽度
        :param new_height: 新的高度
        :param show_step: 是否显示每一步的缩放过程图片
        :return: 缩放后的图片数据
        """
        # 将图片转换为灰度图
        print(f'开始执行像素裁剪,从{image.shape[1]},{image.shape[0]} 裁剪至 {new_width},{new_height}')
        now = datetime.now()
        gray_image = self.convert_to_grayscale(image)
        if show_step:
            pi.plot_image(gray_image, gray=True, title='灰度图', folder=self.name, file_name='1_grey.png')
        # 计算能量
        energy_image = self.calculate_energy(gray_image)

        if show_step:
            pi.plot_image(energy_image, title='能量图', folder=self.name, file_name='2_energy.png')

        current_axis = 1 if image.shape[1] > new_width else 0
        # 计算累计能量
        cumulative_energy = self.calculate_cumulative_energy(energy_image, axis=current_axis)

        if show_step:
            pi.plot_image(cumulative_energy, title='累积能量图', folder=self.name, file_name='3_cumulative_energy.png')

        # 计算最小能量路径
        seam = self.calculate_seam(cumulative_energy, axis=current_axis)

        if show_step:
            marked_image = mark_pixel(image, seam)
            pi.plot_image(marked_image, title='原图中标记裁剪像素的图', folder=self.name, file_name='4_marked.png')
        # 移除最小能量路径
        new_image = self.remove_seam(image, seam)

        if show_step:
            pi.plot_image(new_image, title='裁剪后的图', folder=self.name, file_name='5_new_image.png')

        print(
            f'执行完毕,从{image.shape[1]},{image.shape[0]} 裁剪至 {new_width},{new_height}, 共计时长{datetime.now() - now}')
        pi.cv_save_image(new_image, folder=path.join(self.name, 'list'),
                         file_name=f'{new_image.shape[1]},{new_image.shape[0]}.png')
        # 递归缩放
        if new_width < new_image.shape[1] or new_height < new_image.shape[0]:
            return self.seam_carving(new_image.astype(np.uint8), new_width, new_height)
        else:
            return new_image

    def test_job(self):
        h = self.image.shape[0]
        w = self.image.shape[1]
        self.seam_carving(self.image, w - 1, h, show_step=True)

    def normal_job(self, new_width, new_height):
        self.seam_carving(self.image, new_width, new_height)


# 能量计算简易版(这里计算能量主要是用当前像素的灰度值与上下左右4个像素的灰度值的差值的均值)
class SeamCarvingSimple(SeamCarvingBase):
    name = 'simple'

    def calculate_energy(self, gray_image) -> np.ndarray:
        energy_image = np.zeros((gray_image.shape[0], gray_image.shape[1]))
        gray_image = gray_image.astype(np.int32)
        for i in range(0, gray_image.shape[0]):
            for j in range(0, gray_image.shape[1]):
                data = []
                if i > 0:
                    data.append(abs(gray_image[i - 1, j] - gray_image[i, j]))
                if i < gray_image.shape[0] - 1:
                    data.append(abs(gray_image[i + 1, j] - gray_image[i, j]))
                if j > 0:
                    data.append(abs(gray_image[i, j - 1] - gray_image[i, j]))
                if j < gray_image.shape[1] - 1:
                    data.append(abs(gray_image[i, j + 1] - gray_image[i, j]))

                energy_image[i, j] = np.mean(data)
        return energy_image

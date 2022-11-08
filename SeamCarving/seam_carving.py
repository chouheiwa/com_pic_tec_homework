import os
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
        image_copy[i, seam[i]] = [255, 0, 0]
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

    def rotate_image(self, image, clockwise):
        """
        旋转图片
        :param image: 图片矩阵
        :param clockwise: 顺时针
        :return: 旋转后的图片矩阵
        """
        k = 1 if clockwise else 3
        return np.rot90(image, k)

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

    def calculate_cumulative_energy(self, energy_image):
        """
        计算累计能量
        :param energy_image: 能量数据
        :return: 累计能量数据
        """
        cumulative_energy = np.zeros((energy_image.shape[0], energy_image.shape[1]))

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
        return cumulative_energy

    def calculate_seam(self, cumulative_energy):
        """
        通过累计能量计算能量最低的路径
        :param cumulative_energy: 累计能量图
        :return:
        """
        seam = np.zeros((cumulative_energy.shape[0],), dtype=np.dtype('i4'))
        last_index = cumulative_energy.shape[0] - 1
        for i in range(last_index, -1, -1):
            if i == last_index:
                seam[i] = np.argmin(cumulative_energy[i, :])
            else:
                if seam[i + 1] == 0:
                    seam[i] = seam[i + 1] + np.argmin(cumulative_energy[i, seam[i + 1]:seam[i + 1] + 2])
                elif seam[i + 1] == cumulative_energy.shape[1] - 1:
                    seam[i] = np.argmin(cumulative_energy[i, seam[i + 1] - 1:seam[i + 1] + 1]) + seam[i + 1] - 1
                else:
                    seam[i] = np.argmin(cumulative_energy[i, seam[i + 1] - 1:seam[i + 1] + 2]) + seam[i + 1] - 1
        return seam

    def remove_seam(self, image, seam):
        """
        移除缝
        :param image: 图片源数据
        :param seam: 缝数据
        :return: 移除缝后的图片数据
        """
        image_list = image.tolist()
        removed_pixels = []
        for i in range(len(seam)):
            removed_pixels.append(image_list[i].pop(seam[i]))
        return np.array(image_list).astype(np.uint8), removed_pixels

    def insert_seam(self, image, seam, removed_pixel):
        """
        插入缝
        :param image: 图片源数据
        :param seam: 缝数据
        :param removed_pixel: 移除缝的像素数据
        :return: 插入缝后的图片数据
        """
        h, w = image.shape[:2]
        output = np.zeros((h, w + 1, image.shape[2]))
        for row in range(h):
            col = seam[row]
            for ch in range(3):
                if col == 0:
                    p = np.average(image[row, col: col + 2, ch])
                    output[row, col, ch] = image[row, col, ch]
                    output[row, col + 1, ch] = p
                    output[row, col + 1:, ch] = image[row, col:, ch]
                else:
                    p = np.average(image[row, col - 1: col + 1, ch])
                    output[row, : col, ch] = image[row, : col, ch]
                    output[row, col, ch] = p
                    output[row, col + 1:, ch] = image[row, col:, ch]

        return output

    def process(self, im, gray_im, name, rotated=False, show_step=False, all_step_image=False):
        # 计算能量
        energy_image = self.calculate_energy(gray_im)

        if show_step:
            pi.plot_image(energy_image if not rotated else self.rotate_image(energy_image, False), title='能量图',
                          folder=path.join(self.name, name), file_name='2_energy.png')

        # 计算累计能量
        cumulative_energy = self.calculate_cumulative_energy(energy_image)

        if show_step:
            pi.plot_image(cumulative_energy, title='累积能量图', folder=path.join(self.name, name),
                          file_name='3_cumulative_energy.png')

        # 计算最小能量路径
        seam = self.calculate_seam(cumulative_energy)
        if show_step:
            marked_image = mark_pixel(im, seam)
            pi.plot_image(marked_image if not rotated else self.rotate_image(marked_image, False),
                          title='原图中标记裁剪像素的图', folder=path.join(self.name, name), file_name='4_marked.png')

        if all_step_image:
            marked_image = mark_pixel(im, seam)
            pi.cv_save_image(marked_image if not rotated else self.rotate_image(marked_image, False),
                             folder=path.join(self.name, name, 'list'),
                             file_name=f'{marked_image.shape[1]},{marked_image.shape[0]}_m.png')
        # 移除最小能量路径
        result_im, removed_pixel = self.remove_seam(im, seam)
        remove_gray_im, _ = self.remove_seam(gray_im, seam)
        if show_step:
            pi.plot_image(result_im if not rotated else self.rotate_image(result_im, False), title='裁剪后的图',
                          folder=path.join(self.name, name), file_name='5_new_image.png')
        return result_im, remove_gray_im, seam, removed_pixel

    def seam_carving_remove(self, image, new_width, new_height, show_step=False, all_step_image=False):
        """
        图像压缩
        :param image: 图片源数据
        :param new_width: 新的宽度
        :param new_height: 新的高度
        :param show_step: 是否显示每一步的缩放过程图片
        :param all_step_image: 是否显示中间过程的图片
        :return: 缩放后的图片数据
        """
        # 将图片转换为灰度图

        gray_image = self.convert_to_grayscale(image)

        pi.plot_image(gray_image, gray=True, title='灰度图', folder=self.name, file_name='1_grey.png')

        result_image = image.copy()
        while gray_image.shape[1] > new_width:
            h, w = gray_image.shape[:2]
            print(
                f'开始使用{self.name}执行像素裁剪,从{w},{h} 裁剪至 {w - 1},{h}')
            now = datetime.now()
            result_image, gray_image, seam, _ = self.process(result_image, gray_image, 'remove',
                                                             rotated=False,
                                                             show_step=show_step,
                                                             all_step_image=all_step_image)
            pi.cv_save_image(result_image, folder=path.join(self.name, 'remove', 'list'),
                             file_name=f'{result_image.shape[1]},{result_image.shape[0]}.png')

            print(
                f'执行完毕,从{w},{h} 裁剪至 {w - 1},{h}, 共计时长{datetime.now() - now}')

        result_image = self.rotate_image(result_image, True)
        gray_image = self.rotate_image(gray_image, True)

        while gray_image.shape[1] > new_height:
            h, w = gray_image.shape[:2]
            print(
                f'开始使用{self.name}执行像素裁剪,从{h},{w} 裁剪至 {h - 1},{w}')
            now = datetime.now()
            result_image, gray_image, seam, _ = self.process(result_image, gray_image, 'remove',
                                                             rotated=True,
                                                             show_step=show_step,
                                                             all_step_image=all_step_image)
            pi.cv_save_image(self.rotate_image(result_image, False), folder=path.join(self.name, 'list'),
                             file_name=f'{result_image.shape[1]},{result_image.shape[0]}.png')
            print(
                f'执行完毕,从{h},{w} 裁剪至 {h - 1},{w}, 共计时长{datetime.now() - now}')

    def seam_carving_add(self, image, new_width, new_height, show_step=False, all_step_image=False):
        """
        图像放大
        :param image: 图片源数据
        :param new_width: 新的宽度
        :param new_height: 新的高度
        :param show_step: 是否显示每一步的缩放过程图片
        :param all_step_image: 是否显示中间过程的图片
        :return: 放大后的图片数据
        """
        # 将图片转换为灰度图
        gray_image = self.convert_to_grayscale(image)

        pi.plot_image(gray_image, gray=True, title='灰度图', folder=self.name, file_name='1_grey.png')

        result_image = image.copy()

        k_x = new_width - image.shape[1]
        # k_y = new_height - image.shape[0]
        seams = []
        pixels = []
        while k_x > 0:
            print(
                f'开始使用{self.name}执行像素添加,开始获取第{len(seams) + 1}条最小能量路径')
            now = datetime.now()
            result_image, gray_image, seam, pixel = self.process(result_image, gray_image, 'add',
                                                                 rotated=False,
                                                                 show_step=show_step,
                                                                 all_step_image=all_step_image)
            seams.append(seam)
            pixels.append(pixel)
            pi.cv_save_image(result_image, folder=path.join(self.name, 'add', 'list'),
                             file_name=f'{result_image.shape[1]},{result_image.shape[0]}.png')
            print(
                f'执行完毕，第{len(seams)}条最小能量路径获取完毕, 共计时长{datetime.now() - now}')
            k_x -= 1
        seams.reverse()
        pixels.reverse()
        # pixels = np.array(pixels)
        result_image = image
        for i in range(len(seams) - 1, -1, -1):
            print(
                f'开始使用{self.name}执行像素添加,开始添加{i + 1}能量路径')
            seam = seams.pop()
            pixel = pixels.pop()
            result_image = self.insert_seam(result_image, seam, pixel)
            pi.cv_save_image(result_image, folder=path.join(self.name, 'add', 'list'),
                             file_name=f'{result_image.shape[1]},{result_image.shape[0]}_add.png')
            print(
                f'{self.name}像素添加，{i + 1}能量路径添加完毕')
            for remaining_seam in seams:
                remaining_seam[np.where(remaining_seam >= seam)] += 2

    def test_job(self):
        h = self.image.shape[0]
        w = self.image.shape[1]
        self.seam_carving_add(self.image, w + 1, h, show_step=True, all_step_image=True)
        self.seam_carving_remove(self.image, w - 1, h, show_step=True)

    def normal_job(self, new_width, new_height, all_step_image=False):
        start_time = datetime.now()
        self.seam_carving_remove(self.image, new_width, new_height, all_step_image=all_step_image)
        print(f'{self.name}删除执行完毕,共计时长{datetime.now() - start_time}')

    def add_job(self, new_width, new_height, all_step_image=False):
        start_time = datetime.now()
        self.seam_carving_add(self.image, new_width, new_height, all_step_image=all_step_image)
        print(f'{self.name}添加执行完毕,共计时长{datetime.now() - start_time}')

    def generate_gif(self, name):
        base_dir = path.join('images', 'output', self.name, name)

        list_dir = path.join(base_dir, 'list')
        try:
            files = pi.cv_get_files(list_dir)

            from image_to_gif import image_to_gif
            image_to_gif(files, path.join(base_dir, f'{self.name}_{name}_result.gif'),
                         time=0.08)
            print('生成gif成功，保存在', path.join(base_dir, f'{self.name}_result.gif'))
        except:
            pass



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


class SeamCarvingGradientMagnitude(SeamCarvingBase):
    name = 'gradient_magnitude'

    def calculate_energy(self, gray_image) -> np.ndarray:
        # 计算梯度
        gradient_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        # 计算梯度幅值
        gradient_magnitude = np.sqrt(np.square(gradient_x) + np.square(gradient_y))
        return gradient_magnitude


class SeamCarvingForwardEnergy(SeamCarvingBase):
    name = 'forward_energy'

    def calculate_energy(self, gray_image) -> np.ndarray:
        """
        Forward energy algorithm as described in "Improved Seam Carving for Video Retargeting"
        by Rubinstein, Shamir, Avidan.
        Vectorized code adapted from
        https://github.com/axu2/improved-seam-carving.
        """
        h, w = gray_image.shape

        energy = np.zeros((h, w))
        m = np.zeros((h, w))

        U = np.roll(gray_image, 1, axis=0)
        L = np.roll(gray_image, 1, axis=1)
        R = np.roll(gray_image, -1, axis=1)

        cU = np.abs(R - L)
        cL = np.abs(U - L) + cU
        cR = np.abs(U - R) + cU

        for i in range(1, h):
            mU = m[i - 1]
            mL = np.roll(mU, 1)
            mR = np.roll(mU, -1)

            mULR = np.array([mU, mL, mR])
            cULR = np.array([cU[i], cL[i], cR[i]])
            mULR += cULR

            argmins = np.argmin(mULR, axis=0)
            m[i] = np.choose(argmins, mULR)
            energy[i] = np.choose(argmins, cULR)

        return energy

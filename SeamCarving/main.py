import numpy as np

import SeamCarving.seam_carving as sc
import plot_image as pi

if __name__ == '__main__':
    pic = 'images/test.png'

    seam_carving_list = [
        sc.SeamCarvingSimple(pic),  # 使用简单能量矩阵
        sc.SeamCarvingGradientMagnitude(pic),  # 使用梯度能量矩阵
        sc.SeamCarvingForwardEnergy(pic)  # 使用前向能量矩阵
    ]

    for seam_carving in seam_carving_list:
        seam_carving.test_job()  # 打印测试单步结果
        seam_carving.normal_job(300, 426, all_step_image=True)  # 通用裁剪结果
        seam_carving.add_job(800, 426, all_step_image=True)  # 放大结果
        pass

    pi.cv_save_files()

    for seam_carving in seam_carving_list:
        seam_carving.generate_gif('add')  # 生成放大过程gif动图(此过程在作业提交代码中无效果)
        seam_carving.generate_gif('remove')  # 生成裁剪过程gif动图(此过程在作业提交代码中无效果)

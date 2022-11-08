import numpy as np

import SeamCarving.seam_carving as sc
import plot_image as pi

if __name__ == '__main__':
    pic = 'images/test.png'

    seam_carving_list = [
        sc.SeamCarvingSimple(pic),
        sc.SeamCarvingGradientMagnitude(pic),
        sc.SeamCarvingForwardEnergy(pic)
    ]

    for seam_carving in seam_carving_list:
        seam_carving.test_job()
        seam_carving.normal_job(300, 426, all_step_image=True)
        seam_carving.add_job(800, 426, all_step_image=True)
        pass

    pi.cv_save_files()

    for seam_carving in seam_carving_list:
        seam_carving.generate_gif('add')
        seam_carving.generate_gif('remove')

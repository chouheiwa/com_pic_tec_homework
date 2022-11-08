import imageio
import numpy as np


def image_to_gif(files, output_path, time):
    frames = []
    for i in range(len(files)):
        print('process:[{}]/[{}]'.format(i + 1, len(files)))
        frames.append(imageio.v2.imread(files[i]))
    max_h = 0
    max_w = 0
    for i in range(len(frames)):
        h, w = frames[i].shape[:2]
        max_h = max(max_h, h)
        max_w = max(max_w, w)

    for i in range(len(frames)):
        h, w = frames[i].shape[:2]
        if h < max_h or w < max_w:
            frames[i] = np.pad(frames[i], ((0, max_h - h), (0, max_w - w), (0, 0)),
                               'constant',
                               constant_values=255)

    imageio.v2.mimsave(uri=output_path, ims=frames, format='GIF', duration=time)

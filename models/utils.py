import os

import cv2
from PIL import Image
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
import pickle as pkl


def plot_multi_loss(model_dir):
    import glob
    hist_pkls = glob.glob(os.path.join(model_dir, "history_*.pkl"))
    y1 = []
    y2 = []
    for hist_pkl in hist_pkls:
        datas = pkl.load(open(hist_pkl, 'rb'))
        y1 += datas["D_loss"]
        y2 += datas["G_loss"]

    x = range(len(y1))
    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Iter')
    plt.ylabel('Loss')

    plt.legend(loc="upper right")
    plt.grid(True)
    plt.tight_layout()
    save_path = os.path.join(os.path.split(hist_pkl)[0], "loss.png")
    plt.savefig(save_path)

    plt.close()

def save_images(images, size, image_path):
    return imsave(images, size, image_path)

def imsave(images, size, path):
    image = np.squeeze(merge(images, size))
    return scipy.misc.imsave(path, image)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3,4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3]==1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter ''must have dimensions: HxW or HxWx3 or HxWx4')

def pil_loader(image_path):
    img = Image.open(open(image_path, 'rb'))
    return img.convert('RGB')

def read_mask(mask_path, new_h, new_w):
    """[summary]

    Args:
        mask_path ([type]): [path to mask directory]
        new_h ([type]): [new height of image]
        new_w ([type]): [new weight of image]

    Returns:
        [type]: [description]
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (new_w, new_h))
    ret, mask = cv2.threshold(mask, 127, 255, 0)
    res = np.zeros([new_h, new_w, 3], np.float32)
    res[:, :, 0] = mask
    res[:, :, 1] = mask
    res[:, :, 2] = mask

    return res / 255
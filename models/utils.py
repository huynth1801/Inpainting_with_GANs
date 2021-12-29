import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import convolve2d 
import cv2
from PIL import Image
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
import pickle as pkl

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


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

def create_mask_weight(mask, n_size):
    kernel = np.ones((n_size, n_size), dtype=float)
    kernel = kernel / np.sum(kernel)
    mask_weight = np.zeros(mask.shape, dtype=float)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            mask_weight[i, j] = convolve2d(mask[i, j], kernel, mode='same', boundary='symm' )
    mask_weight = mask * ( 1.0 - mask_weight )
    return torch.FloatTensor(mask_weight)

class ContextLoss(nn.Module):
    def __init__(self):
        super(ContextLoss, self).__init__()
        self.l1_loss = nn.L1Loss()

    def forward(self, gen_imgs, masked_imgs, mask_weight):
        loss = self.l1_loss(mask_weight*gen_imgs, mask_weight*masked_imgs)
        return loss

class PriorLoss(nn.Module):
    # Prior loss: Lp = lamda * log(1- D(G(z)))
    def __init__(self):
        super(PriorLoss, self).__init__()
    
    def forward(self, lamda, d_gen_imgs):
        ones = torch.ones_like(d_gen_imgs)
        loss = lamda * torch.log(ones - d_gen_imgs)
        loss = torch.mean(loss)
        return loss

def image_gradient(img):
    a = torch.FloatTensor([[[[1, 0, -1],
                             [2, 0, -2],
                             [1, 0, -1]],
                            [[1, 0, -1],
                             [2, 0, -2],
                             [1, 0, -1]],
                            [[1, 0, -1],
                             [2, 0, -2],
                             [1, 0, -1]]
                            ]]).to(device)
    G_x = F.conv2d(img, a, padding=1)
    b = torch.FloatTensor([[[[1, 2, 1],
                             [0, 0, 0],
                             [-1, -2, -1]],
                            [[1, 2, 1],
                             [0, 0, 0],
                             [-1, -2, -1]],
                            [[1, 2, 1],
                             [0, 0, 0],
                             [-1, -2, -1]]
                            ]]).to(device)
    G_y = F.conv2d(img, b, padding=1)
    return G_x, G_y

def posisson_blending(mask, gen_imgs, masked_imgs, args):

    x = mask * masked_imgs + (1 - mask) * gen_imgs

    x_optimum = nn.Parameter(
        torch.FloatTensor(x.detach().cpu().numpy()).to(device))

    optimizer_inpaint = torch.optim.Adam([x_optimum], lr=args.lr, betas=(args.b1, args.b2))
    gen_imgs_G_x, gen_imgs_G_y = image_gradient(gen_imgs)

    for epoch in range(args.blending_steps):

        optimizer_inpaint.zero_grad()

        x_G_x, x_G_y = image_gradient(x_optimum)
        blending_loss = torch.sum((x_G_x - gen_imgs_G_x) ** 2 + (x_G_y - gen_imgs_G_y) ** 2)

        blending_loss.backward()

        optimizer_inpaint.step()

        print("[Epoch: {}/{}]\tBlending loss: {:4f}".format(epoch, args.blending_steps,  blending_loss.item()))

    return x_optimum.detach()

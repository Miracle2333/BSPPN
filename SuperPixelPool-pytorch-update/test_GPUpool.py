from __future__ import print_function
import torch
from suppixpool_layer import AveSupPixPool, MaxSupPixPool, SupPixUnpool
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import time
from skimage.segmentation import slic
from skimage import io
from skimage.segmentation import mark_boundaries
from skimage.color import rgb2gray
import matplotlib.pyplot as plt



if __name__ == "__main__":

    img1 = io.imread('RGB-PanSharpen_AOI_3_Paris_img829.tif')
    # img1 = rgb2gray(img1)
    # img = img1[500:600,500:600,1]
    GPU = torch.device("cuda")
    batch_size = 1
    n_channels = 1
    xSize = np.size(img1, 0)
    ySize = np.size(img1, 1)



    # X = torch.randn((batch_size,n_channels,xSize,ySize), dtype=torch.float32, device=GPU, requires_grad=True)

    X = np.array([img1[:,:,0].reshape(n_channels, xSize, ySize)] * batch_size, dtype=np.uint8)

    X = X / 255
    spx = np.array([np.arange(xSize * ySize).reshape(xSize, ySize)] * batch_size)
    sp_input = img1
    spx[0] = slic(sp_input, n_segments=50)
    sp = spx[0,:,:]
    segments_boundary = mark_boundaries(sp_input, sp)
    # X = X*255
    # fig = plt.figure("Superpixel")
    # ax = fig.add_subplot(2, 1, 1)
    # ax.imshow(segments_boundary)
    # ax = fig.add_subplot(2, 1, 2)
    # ax.imshow(sp_input)
    # plt.show()

    # spx = np.array([np.arange(xSize*ySize).reshape(xSize,ySize)]*batch_size)
    # spx = np.zeros((batch_size, xSize, ySize))
    # # spx[0] = spx[0].astype(np.uint8)
    # spx[0] = slic(spx[0], n_segments=10)
    # spx[0] = spx[0].astype(np.float)
    # spx = torch.tensor(spx, dtype=torch.float32, requires_grad=True)

    X = torch.tensor(X, dtype=torch.float32, device=GPU, requires_grad=True)
    spx = torch.tensor(spx, device=GPU)
    # spx = spx.to(GPU)
    # X.detach()
    # X + X
    # print ("INPUT ARRAY  ----------------- \n", X)
    # with torch.enable_grad():
    pool = MaxSupPixPool()
    pld = pool(X, spx)
    #     loss = pld.sum()
    # loss.backward()
    # print(X.grad)
    # print ("POOLED ARRAY ----------------- \n", pld)
    # print ("Shape of pooled array: ", pld.size())
    unpool = SupPixUnpool()
    unpld = unpool(pld, spx)
    # # a = (unpld == X)
    # # print ("Unpooling back to original: ", (a.data).np.all(axis=none, keepdims=true))

    # X = Variable(X, requires_grad=True)
    # spx = Variable(spx, requires_grad=False)

    res = torch.autograd.gradcheck(pool, (X, spx), raise_exception=False)
    resUnpool = torch.autograd.gradcheck(unpool, (pld, spx), raise_exception=False)

    print ("Gradients of pooling are {}.".format("correct" if res else "wrong")) # res should be True if the gradients are correct.
    print ("Gradients of unpooling are {}.".format("correct" if resUnpool else "wrong"))
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
import os
from torch.autograd import Variable
import matplotlib.colors as colors
import cv2
import math

def conv(batchnorm, input, output, kernel=7, stride=1):
    if batchnorm:
        return nn.Sequential(nn.Conv2d(input, output, kernel_size=kernel, stride=stride, padding=(kernel - 1) // 2),
                             nn.BatchNorm2d(output),
                             nn.LeakyReLU(0.1, inplace=False))
    else:
        return nn.Sequential(nn.Conv2d(input, output, kernel_size=kernel, stride=stride, padding=(kernel - 1) // 2),
                             nn.LeakyReLU(0.1, inplace=False))


def basic_module(batchnorm):
    return nn.Sequential(conv(batchnorm=batchnorm, input=8, output=32,kernel=5,stride=1),
                         conv(batchnorm=batchnorm,input=32,output=64,kernel=5,stride=1),
                         conv(batchnorm=batchnorm,input=64,output=128,kernel=5,stride=1),
                         conv(batchnorm=batchnorm,input=128,output=64,kernel=5,stride=1),
                         conv(batchnorm=batchnorm,input=64,output=32,kernel=5,stride=1),
                         )


def bilinear_interpolation(source, out_dim):
    channel, src_h, src_w, = source.shape[-3:]
    dst_h, dst_w = out_dim[0], out_dim[1]
    if src_h == dst_h and src_w == dst_w:
        return source.copy()
    dst = torch.zeros((channel, dst_h, dst_w), dtype=torch.float)
    scale_y, scale_x = float(src_w) / dst_w, float(src_h) / dst_h
    for i in range(channel):
        for dst_x in range(dst_h):
            for dst_y in range(dst_w):
                # find the origin x and y coordinates of dst image x and y
                # use geometric center symmetry
                # if use direct way, src_x = dst_x * scale_x
                src_x = (dst_x + 0.5) * scale_x - 0.5
                src_y = (dst_y + 0.5) * scale_y - 0.5

                # find the coordinates of the points which will be used to compute the interpolation
                src_x0 = int(np.floor(src_x))
                src_x1 = min(src_x0 + 1, src_w - 1)
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, src_h - 1)

                # calculate the interpolation
                temp0 = (src_x1 - src_x) * source[...,i, src_x0, src_y0] + (src_x - src_x0) * source[...,i, src_x0, src_y1]
                temp1 = (src_x1 - src_x) * source[...,i, src_x0, src_y1] + (src_x - src_x0) * source[...,i, src_x1, src_y1]
                dst[dst_y, dst_x, i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)

    return dst

def warp(x, flo):
        B, C, H, W = x.size()
        # mesh grid
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()

        if x.is_cuda:
            grid = grid.cuda()
        vgrid = Variable(grid) + flo

        
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)
        output = nn.functional.grid_sample(x, vgrid)
        mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
        mask = nn.functional.grid_sample(mask, vgrid)


        mask[mask < 0.9999] = 0
        mask[mask > 0] = 1

        return output * mask

def charbonnier_loss(delta, alpha=0.45, epsilon=1e-5):
    cha=(delta**2. +1e-5**2.)**alpha
    return torch.mean(cha)

def CFM(event,flow):
    tmax=(torch.max(event)-torch.min(event)).item()
    t_ref=torch.min(event).item()
    warped=torch.zeros_like(event)
    for k in range(event.shape[0]):
        for i in range(event.shape[2]):
            for j in range(event.shape[3]):
                x_warped=(event[i,j]-t_ref)*flow[i,j,0]+i
                y_warped=(event[i,j]-t_ref)*flow[i,j,0]+j
                if (x_warped>0 and x_warped<event.shape[0]) and (y_warped>0 and y_warped<event.shape[1]):
                    warped[np.floor(x_warped),np.floor(y_warped)]+=1
    h=torch.mean((warped-torch.mean(warped))**2)
    return h

def gradient(x):
    dy=x.new_zeros(x.shape)
    dx=x.new_zeros(x.shape)

    dy[...,1:,:]=x[...,1:,:]-x[...,:-1,:]
    dx[...,:,1:]=x[...,:,1:]-x[...,:,:-1]
    return dx,dy

def Smoothness_loss(flow):
    dx,dy=gradient(flow)

    loss=charbonnier_loss(dx)+charbonnier_loss(dy)

    return loss/2.


def Photometric_loss(flow,image):
    loss = 0
    for i in range(len(flow)):
        warped = warp(image[i][1].float().cuda(), flow[i])
        loss += charbonnier_loss(warped - image[i][0].float().cuda())
    return loss/(len(flow))

def All_loss(flow,image,event=None):
    loss=Photometric_loss(flow,image)
    for f in flow:
        loss+=0.5*Smoothness_loss(f)
    #for (f,event) in zip(flow,event):
        #loss+=CFM(event,f)
    return loss

def ArrayToTensor(array):
    assert type(array) is np.ndarray
    return torch.from_numpy(array)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return '{:.3f}({:.3f})'.format(self.val, self.avg)


def save_checkpoint(state, save_path, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(save_path, filename))





def single_flow2rgb(flow_x, flow_y, hsv_buffer=None):
    if hsv_buffer is None:
        hsv_buffer = np.empty((flow_x.shape[0],flow_x.shape[1],3))
    hsv_buffer[:,:,1] = 1.0
    hsv_buffer[:,:,0] = 360*(np.arctan2(flow_y,flow_x)+np.pi)/(2.0*np.pi)

    hsv_buffer[:,:,2] = np.linalg.norm( np.stack((flow_x,flow_y), axis=0), axis=0 )

    flat = hsv_buffer[:,:,2].reshape((-1))
    m = np.nanmax(flat[np.isfinite(flat)])
    if not np.isclose(m,0.0):
        hsv_buffer[:,:,2] /= m

    return colors.hsv_to_rgb(hsv_buffer)


def flow_viz_np(flow_x,flow_y):
   
    flows = np.stack((flow_x, flow_y), axis=2)
    mag = np.linalg.norm(flows, axis=2)

    ang = np.arctan2(flow_y,flow_x)
    ang += np.pi
    ang *= 180. / np.pi / 2.
    ang = ang.astype(np.uint8)
    hsv = np.zeros([flow_x.shape[0], flow_x.shape[1], 3], dtype=np.uint8)
    hsv[:, :, 0] = ang
    hsv[:, :, 1] = 255
    hsv[:, :, 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return flow_rgb

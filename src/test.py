from MODEL import *
import time
import argparse, os, time
import torch
import random
import numpy as np
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from utils import *
import cv2


if __name__=="__main__":
    model = CRGCANet_V4()
    model = nn.DataParallel(model).cuda()

    k=cv2.resize(cv2.imread(r'/media/xusper/KESU/DEHAZE/outdoor/indoor/nyuhaze500/hazy/1400_1.png'),(300,300))
    input = torch.unsqueeze(ArrayToTensor(k).permute(2,0,1),0).float().cuda()
    start=time.time()
    o = model(input)
    print(time.time()-start)

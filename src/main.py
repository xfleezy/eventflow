import argparse
import os
import time

import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from utils import *
import MODEL
from tensorboardX import SummaryWriter
import dataset
import datetime
import cv2
import config
def main():
    args = configs()
    timestamp = datetime.datetime.now().strftime("%m-%d-%H:%M")
    save_path = os.path.join(timestamp, args.save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    train_writer = SummaryWriter(os.path.join(save_path, 'train'))
    test_writer = SummaryWriter(os.path.join(save_path, 'test'))

    input_transform = transforms.Compose([])
    print("=> fetch img pairs in {}".format("event_data"))
    train_set = dataset.PyDataset(args.event_path,args.gray_path)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    if args.pretrained:
        network_data = torch.load(args.pretrained)
    else:
        network_data = None
        print("=> creat model: pyramid-net")
    model = MODEL.pyramidnet(network_data).cuda()
    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True

    param_groups = [{'params': model.module.bias_parameters(), 'weight_decay': 0},
                    {'params': model.module.weight_parameters(), 'weight_decay': args.weight_decay}]

    optimizer = torch.optim.Adam(param_groups, args.lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestone, gamma=args.gamma)
    for epoch in range(args.epoch):
        scheduler.step()
        loss = train(train_loader, model, optimizer, epoch, train_writer,args.iter)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': "pyramide",
            'state_dict': model.module.state_dict(), 'loss': loss
        }, save_path)

def train(train_loader, model, optimizer, epoch, train_writer,n_iter):
    batch_time = AverageMeter()
    date_time = AverageMeter()
    losses = AverageMeter()
    epoch_size = len(train_loader)

    model.train()
    end = time.time()

    for i, (event,gray,image) in enumerate(train_loader):
        flow = model(event,gray)
        
        loss = All_loss(flow,image)

        losses.update(loss.item())
        train_writer.add_scalar('trainloss', loss.item(), n_iter)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()
        n_iter += 1
        if i%n_iter==0:
            print(i,":",losses.avg)
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    print('epoch',epoch,":",losses.avg,'/',)
    return losses.avg


def validate(model, epoch, eval_loader, output_writer=None):
    global args
    batch_time = AverageMeter()
    Error=AverageMeter()
    model.eval()
    end = time.time()
    if not os.path.exists(out_path+str(epoch)):
        os.makedirs(out_path+str(epoch))

    for i, (event,gray,image) in enumerate(eval_loader):
            flow = model(event,gray).detach().cpu().numpy()
            flow=np.squeeze(flow,axis=0).transpose(1,2,0)
            flow= flow_viz_np(flow[:,:,0],flow[:,:,1])
            cv2.imwrite(os.path.join(out_path+str(epoch),str(i)+'.jpg'),flow)


if __name__=="__main__":
    main()

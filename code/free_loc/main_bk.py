import argparse
import os
import shutil
import time
import sys
# sys.path.insert(0,'/home/spurushw/reps/hw-wsddn-sol/faster_rcnn')
sys.path.insert(0, '/home/ubuntu/aws_share/sol/hw2-solution/code/faster_rcnn/')
sys.path.insert(0, '/home/ubuntu/aws_share/sol/hw2-solution/code/')
import sklearn
import sklearn.metrics

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import visdom
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score

import logger
from datasets.factory import get_imdb
from custom import *

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# parser.add_argument('--arch', default='localizer_alexnet')
parser.add_argument('--arch', default='localizer_alexnet_robust')

parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--eval-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
# parser.add_argument('--resume', default='', type=str, metavar='PATH',
#                     help='path to latest checkpoint (default: none)')
parser.add_argument('--resume', default='', type=str, metavar='./checkpoints',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model', default=True)
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--vis',action='store_true')

best_prec1 = 0


def main():
    global args, best_prec1
    torch.manual_seed(157)
    torch.cuda.manual_seed(157)
    #np.random.seed(0)
    args = parser.parse_args()
    args.distributed = args.world_size > 1

    # create model
    print("=> creating model '{}'".format(args.arch))
    if args.arch=='localizer_alexnet':
        model = localizer_alexnet(pretrained=args.pretrained)
    elif args.arch=='localizer_alexnet_robust':
        model = localizer_alexnet_robust(pretrained=args.pretrained)
    print(model)

    # torch.cuda.set_device(0)
    # print(torch.cuda.current_device())
    model.features = torch.nn.DataParallel(model.features)
    model.cuda()

    # TODO:
    # define loss function (criterion) and optimizer
    criterion = nn.MultiLabelSoftMarginLoss().cuda()
    # criterion = nn.BCEWithLogitsLoss().cuda()
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    optimizer = torch.optim.SGD(model.classifier.parameters(), args.lr,
            momentum = args.momentum,
            weight_decay = args.weight_decay)
    
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    # TODO: Write code for IMDBDataset in custom.py
    # trainval_imdb.num_images = 5011
    trainval_imdb = get_imdb('voc_2007_trainval')
    test_imdb = get_imdb('voc_2007_test')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_dataset = IMDBDataset(
        trainval_imdb,
        transforms.Compose([
            transforms.Resize((512,512)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    train_sampler = None
    # len(train_loader.dataset.imgs) = 5011
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    # len(val_loader.dataset.imgs) = 4952, dir(val_loader) = 'class_to_idx', 'classes', 'imdb', 'imgs', 'loader', 'target_transform', 'transform'
    val_loader = torch.utils.data.DataLoader(
        IMDBDataset(test_imdb, transforms.Compose([
            transforms.Resize((384,384)),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    
    


    # TODO: Create loggers for visdom and tboard
    # TODO: You can pass the logger objects to train(), make appropriate
    # modifications to train()
    if args.arch == 'localizer_alexnet':
        data_log = logger.Logger('./logs/task1', name='freeloc')
        vis = visdom.Visdom(server='http://localhost', port='8097')
    else:
        data_log = logger.Logger('./logs_robust/task1', name='freeloc')
        vis = visdom.Visdom(server='http://localhost', port='8097')

    if args.evaluate:
        validate(val_loader, model, criterion,vis,0)
        return

    if args.arch == 'localizer_alexnet':
        args.epochs = 30
    else:
        args.epochs = 45
    import pdb;pdb.set_trace()
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, data_log, vis, args.arch)

        # evaluate on validation set
#         if epoch%args.eval_freq==0 or epoch==args.epochs-1:
        if epoch%6==0 or epoch==args.epochs-1:
            print("now evaluating!!!!!!")
            m1, m2 = validate(val_loader, model, criterion,vis,epoch)
            score = m1*m2
            # remember best prec@1 and save checkpoint
            is_best =  score > best_prec1
            best_prec1 = max(score, best_prec1)
            data_log.scalar_summary(tag='valid/metric1',value=m1,step= epoch)
            data_log.scalar_summary(tag='valid/metric2',value=m2,step= epoch)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer' : optimizer.state_dict(),
            }, is_best)

    print('end of training')


#TODO: You can add input arguments if you wish
def train(train_loader, model, criterion, optimizer, epoch, data_log, vis, mode):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    avg_m1 = AverageMeter()
    avg_m2 = AverageMeter()

    # switch to train mode
    model.train()
    
    end = time.time()
    total_iter = len(train_loader)
    upsampler = transforms.Compose([transforms.ToPILImage(),
                                   transforms.Resize((512,512))])
    cm_jet = mpl.cm.get_cmap('jet')
    to_pil = transforms.ToPILImage()
    cnt = 0
    images = np.zeros((1, 512, 512, 3))
    cls_names = train_loader.dataset.classes
    # images_vis = np.zeros((train_loader.batch_size, 512, 512, 3))
    # heatmap_vis = []
    if mode == 'localizer_alexnet':
        last_epoch = 28
    else:
        last_epoch = 43
    # len(train_loader.dataset.imgs) = 5011,dir(train_loader) = 'class_to_idx', 'classes', 'imdb', 'imgs', 'loader', 'target_transform', 'transform'
    # input.shape = torch.Size([32, 3, 512, 512]); target.shape = torch.Size([32, 20])
    for i, (input, target, im_idx) in enumerate(train_loader):

        # len(train_loader) = 157, the number of batches; cur_step = current total batch
        cur_step = epoch * total_iter + i
        # measure data loading time
        data_time.update(time.time() - end)
        # input.shape = torch.Size([32, 3, 512, 512]) target.shape = torch.Size([32, 20])
        target = target.type(torch.FloatTensor).cuda(async=True)
        input_var = torch.autograd.Variable(input, requires_grad=True) #input_var.shape = torch.Size([32, 3, 512, 512])
        target_var = torch.autograd.Variable(target)

        # TODO: Get output from model
        # TODO: Perform any necessary functions on the output
        # TODO: Compute loss using ``criterion``
        # compute output
        input_var = input_var.cuda() # input_var = torch.Size([32, 3, 512, 512])
        output = model(input_var) # output = torch.Size([32, 20, 29, 29])

        # store visualization
        if (i%int(len(train_loader)/4)) == 0 and epoch < 15:
            for j in range(4):
                # index of the ground truth in the target labels, ex: target= [0,1,0,0], gt_indice = 1
                gt_indice = [idx for idx in range(output.shape[1]) if target[j, idx] == 1]
                          
                heatmap = np.zeros((len(gt_indice), 512, 512, 4))
                # img_vis.shape = (3, 512, 512)
                img_vis = input[j].cpu().numpy()
                img_vis = (img_vis - img_vis.min()) / (img_vis.max() - img_vis.min())
                vis.image(img_vis, \
                       opts=dict(title= str(epoch) + '_' + str(i) + '_' + str(j) + '_image'))

                images[0] = np.uint8(np.transpose(img_vis, (1, 2, 0)) * 255)
                a = output.data.cpu().numpy() # output.data = torch.Size([32, 20, 29, 29])
                a.resize((a.shape[0], a.shape[1], 1, a.shape[2], a.shape[3]))
                for cls_i in range(len(gt_indice)):
                    k = gt_indice[cls_i]
                    # a[j,k] (shape torch.Size([29, 29])) get the image which is j image in the batch and k class
                    # normalize the a[j,k] so that the minimum is a[j,k].min()
                    a_norm = (a[j, k] + a[j,k].min()) / (a[j,k].max() - a[j,k].min())

                    # upsampel the image from torch.Size([29, 29]) to (512,512)
                    b = upsampler(torch.Tensor(a_norm))
                    b = np.uint8(cm_jet(np.array(b)) * 255)
                    heatmap[cls_i] = b
                    
                    vis.image(np.transpose(b,(2,0,1)),
                          opts=dict(title= str(epoch) + '_' + str(i) + '_' + str(j) + 'heatmap' + cls_names[k]))
                print(str(cnt + epoch * 4)+' '+str(j))
                data_log.image_summary(tag='/train/'+str(cnt+epoch*4)+'/'+str(j)+'/heatmap',
                                       images=heatmap, step=cur_step)
                data_log.image_summary(tag='/train/'+str(cnt+epoch*4)+'/'+str(j)+'/images',
                                       images=images, step=cur_step)
                
            cnt = cnt+1

        ks = output.size()[2] # ks = 29 output = torch.Size([32, 20, 29, 29])
        global_max = nn.MaxPool2d(kernel_size=(ks, ks)) # max_pool kernel = torch.Size([29, 29])
        output = global_max(output) #in: output.shape = torch.Size([32, 20, 29, 29]), out: output.shape = torch.Size([32, 20, 1, 1])
        output = output.view(output.shape[0], output.shape[1]) # out: output.shape = torch.Size([32, 20])
        loss = criterion(output, target_var) # loss = torch.Size([1])


        # measure metrics and record loss
        m1 = metric1(output.data, target)
        m2 = metric2(output.data, target)
        losses.update(loss.data[0], input.size(0))
        avg_m1.update(m1, input.size(0))
        avg_m2.update(m2, input.size(0))
        
        # TODO: 
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Metric1 {avg_m1.val:.3f} ({avg_m1.avg:.3f})\t'
                  'Metric2 {avg_m2.val:.3f} ({avg_m2.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, avg_m1=avg_m1,
                   avg_m2=avg_m2))

        #TODO: Visualize things as mentioned in handout
        #TODO: Visualize at appropriate intervals
        if i % args.print_freq == 0:
            data_log.scalar_summary(tag='train/loss', value=losses.avg, step=cur_step)
            data_log.scalar_summary(tag='train/top1_metric', value=avg_m1.avg, step=cur_step)
            data_log.scalar_summary(tag='train/top3_metric', value=avg_m2.avg, step=cur_step)
            data_log.model_param_histo_summary(model=model, step=cur_step)

def validate(val_loader, model, criterion,vis,epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    avg_m1 = AverageMeter()
    avg_m2 = AverageMeter()


    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target, im_idx) in enumerate(val_loader):
        target = target.type(torch.FloatTensor).cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)
        
        upsampler = transforms.Compose([transforms.ToPILImage(),
                                       transforms.Resize((384,384))])
        cm_jet = mpl.cm.get_cmap('jet')
        to_pil = transforms.ToPILImage()
        cnt = 0
        images = np.zeros((1, 384, 384, 3))

        # TODO: Get output from model
        # TODO: Perform any necessary functions on the output
        # TODO: Compute loss using ``criterion``
        # compute output
        # target = target.cuda()
        # target_var = target_var.cuda()
        input_var = input_var.cuda()
        output = model(input_var)
        ks = output.size()[2]
        global_max = nn.MaxPool2d(kernel_size=(ks, ks))
        output = global_max(output)
        output = output.view(output.shape[0], output.shape[1])
        loss = criterion(output, target_var)



        # measure metrics and record loss
        m1 = metric1(output.data, target)
        m2 = metric2(output.data, target)
        losses.update(loss.data[0], input.size(0))
        avg_m1.update(m1, input.size(0))
        avg_m2.update(m2, input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Metric1 {avg_m1.val:.3f} ({avg_m1.avg:.3f})\t'
                  'Metric2 {avg_m2.val:.3f} ({avg_m2.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   avg_m1=avg_m1, avg_m2=avg_m2))
            

        #TODO: Visualize things as mentioned in handout
        #TODO: Visualize at appropriate intervals
#         if i%1==0:
#             for j in range(4):
#                 # index of the ground truth in the target labels, ex: target= [0,1,0,0], gt_indice = 1
#                 gt_indice = [idx for idx in range(output.shape[1]) if target[j, idx] == 1]

#                 heatmap = np.zeros((len(gt_indice), 384, 348, 4))
#                 # img_vis.shape = (3, 512, 512)
#                 img_vis = input[j].cpu().numpy()
#                 img_vis = (img_vis - img_vis.min()) / (img_vis.max() - img_vis.min())
#                 vis.image(img_vis, \
#                        opts=dict(title= str(epoch) + '_' + str(i) + '_' + str(j) + '_image'))

#                 images[0] = np.uint8(np.transpose(img_vis, (1, 2, 0)) * 255)
#                 a = output.data.cpu().numpy() # output.data = torch.Size([32, 20, 29, 29])
#                 a.resize((a.shape[0], a.shape[1], 1, a.shape[2], a.shape[3]))
#                 for cls_i in range(len(gt_indice)):
#                     k = gt_indice[cls_i]
#                     # a[j,k] (shape torch.Size([29, 29])) get the image which is j image in the batch and k class
#                     # normalize the a[j,k] so that the minimum is a[j,k].min()
#                     a_norm = (a[j, k] + a[j,k].min()) / (a[j,k].max() - a[j,k].min())

#                     # upsampel the image from torch.Size([29, 29]) to (512,512)
#                     b = upsampler(torch.Tensor(a_norm))
#                     b = np.uint8(cm_jet(np.array(b)) * 255)
#                     heatmap[cls_i] = b

#                     vis.image(np.transpose(b,(2,0,1)),
#                           opts=dict(title= str(epoch) + '_' + str(i) + '_' + str(j) + 'heatmap' + cls_names[k]))




    print(' * Metric1 {avg_m1.avg:.3f} Metric2 {avg_m2.avg:.3f}'
          .format(avg_m1=avg_m1, avg_m2=avg_m2))

    return avg_m1.avg, avg_m2.avg


# TODO: You can make changes to this function if you wish (not necessary)
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def metric1(output, target):
    # TODO: Ignore for now - proceed till instructed
    corr = 0.0 # output.shape = torch.Size([32, 20]): 20 is the max score for each class
    for i in range(target.shape[0]):
        s = output[i]
        sort_idx = sorted(range(len(s)), key=lambda k: s[k]) # sort the output[i] list value from big to small, and return the index of the sorted list from right to left (index 1 == the biggest value, index 19 == the smallest value) ex: output[i] = [10,20,5,8] ->> sort_idx = [2,3,0,1]
        top1_output = np.take(output[i], sort_idx[-1]) # choose the biggest value of the output using the sort_idx[-1] which is the index of the biggest value
        top1_target = np.take(target[i], sort_idx[-1]) # if the index of biggest value in the target is 0, it means that the prediction is wrong, so the multiplication of top1_output*top1_target will be top1_output*0 = 0; in the other hand, if the index of the biggest value is 1, it mean that the prediction is correct, the multiplication of top1_output*top1_target will be top1_output*1 = top1_output
        corr = corr+int((top1_output*top1_target) !=0)*1.0 # corr: accumulate the sum of the correct predicted output score, corr/target.shape[0]: corr divides the batch size to get the average sum of the correct predicted output score

    return corr/target.shape[0]

def metric2(output, target):
    # TODO: Ignore for now - proceed till instructed
#     for i in len(output):
#         out = np.max(output[i])
#         tar = target[i]
#         recall_score(tar,out)
        
    corr = 0.0
    for i in range(target.shape[0]):
        s = output[i]
        sort_idx = sorted(range(len(s)), key=lambda k: s[k])
        top3_output = np.take(output[i], sort_idx[-3:])
        top3_target = np.take(target[i], sort_idx[-3:])
        corr = corr+int((top3_output*top3_target).sum()!=0)*1.0

    return corr/target.shape[0]

if __name__ == '__main__':
    main()

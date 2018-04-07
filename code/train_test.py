from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import _init_paths
import os
import torch
import torch.utils.model_zoo as model_zoo
from torch.nn.parameter import Parameter
import numpy as np
from datetime import datetime

import cPickle as pkl
import network
from wsddn import WSDDN
from utils.timer import Timer
import logger
import roi_data_layer.roidb as rdl_roidb
from roi_data_layer.layer import RoIDataLayer
from datasets.factory import get_imdb
from fast_rcnn.config import cfg, cfg_from_file
from test import test_net
import visdom

try:
    from termcolor import cprint
except ImportError:
    cprint = None

def log_print(text, color=None, on_color=None, attrs=None):
    if cprint is not None:
        cprint(text, color=color, on_color=on_color, attrs=attrs)
    else:
        print(text)

# hyper-parameters
# ------------
imdb_name = 'voc_2007_trainval'
imdb_name_test = 'voc_2007_test'
cfg_file = 'experiments/cfgs/wsddn.yml'
pretrained_model = 'data/pretrained_model/alexnet_imagenet.npy'
output_dir = 'models/saved_model'
visualize = True
vis_interval = 1000
is_plot_loss = True
is_plot_weights = True

start_step = 0
end_step = 50000
lr_decay_steps = {150000}
lr_decay = 1./10

rand_seed = 1024
_DEBUG = False
use_tensorboard = True
use_visdom = True
log_grads = False

remove_all_log = False   # remove all historical experiments in TensorBoard
exp_name = None # the previous experiment name in TensorBoard

thresh = 0.0001

# ------------

if rand_seed is not None:
    np.random.seed(rand_seed)

# load config file and get hyperparameters
cfg_from_file(cfg_file)
lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY
disp_interval = cfg.TRAIN.DISPLAY
log_interval = cfg.TRAIN.LOG_IMAGE_ITERS
if use_tensorboard:
    data_log = logger.Logger('./logs/', name='task2_train')
if use_visdom:
    viz = visdom.Visdom(server='http://localhost', port='8097')
    loss_win = viz.line(X=np.array([0]),Y=np.array([0]), opts=dict(xlable='step', ylabel='Train Loss', title='trainning loss', showlegend=True))
    ap_win = viz.line(X=np.array([0]),Y=np.array([0]), opts = dict(xlabel='step', ylabel='mean AP', title='Test mean AP', showlegend=True))


# load imdb and create data later
imdb = get_imdb(imdb_name)
imdb_test = get_imdb(imdb_name_test)
rdl_roidb.prepare_roidb(imdb)
roidb = imdb.roidb # A roidb is a list of dictionaries, each with the following keys: boxes gt_overlaps gt_classes flipped, len(roidb)=5011s
data_layer = RoIDataLayer(roidb, imdb.num_classes) # generate random permutation, '_get_next_minibatch', '_get_next_minibatch_inds'
 

# Create network and initialize
net = WSDDN(classes=imdb.classes, debug=_DEBUG)
net.features = torch.nn.DataParallel(net.features)
net.roi_pool = torch.nn.DataParallel(net.roi_pool)
net.classifier = torch.nn.DataParallel(net.classifier)
net.score_cls = torch.nn.DataParallel(net.score_cls)
net.score_det = torch.nn.DataParallel(net.score_det)

network.weights_normal_init(net, dev=0.001)
if os.path.exists('pretrained_alexnet.pkl'):
    pret_net = pkl.load(open('pretrained_alexnet.pkl','r'))
else:
    pret_net = model_zoo.load_url('https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth')
    pkl.dump(pret_net, open('pretrained_alexnet.pkl','wb'), pkl.HIGHEST_PROTOCOL)
own_state = net.state_dict() # net.state_dict(),keys = 
for name, param in pret_net.items():
    if name not in own_state:
        continue
    if isinstance(param, Parameter):
        param = param.data
    try:
        own_state[name].copy_(param)
        print('Copied {}'.format(name))
    except:
        print('Did not find {}'.format(name))
        continue
    if 'features' in name:
        name = name.replace('features.','features.module.')
    if 'classifier' in name:
        m = re.search('\d', name)
        k = m.start()
        name = name[:k] + str(int(name[k])-1) + name[k+1:]
        name = name.replace('classifier.', 'classifier.module.')

# Move model to GPU and set train mode
net.cuda()
net.train()


# Create optimizer for network parameters
params = list(net.parameters())
optimizer = torch.optim.SGD(params[2:], lr=lr, 
                            momentum=momentum, weight_decay=weight_decay)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# training
train_loss = 0
tp, tf, fg, bg = 0., 0., 0, 0
step_cnt = 0
re_cnt = False
t = Timer()
t.tic()

for step in range(start_step, end_step+1):
    net.train()
    # get one batch
    blobs = data_layer.forward()
    im_data = blobs['data'] # im_data = (2, 600, 901, 3)
    rois = blobs['rois']
    im_info = blobs['im_info']
    gt_vec = blobs['labels']
    #gt_boxes = blobs['gt_boxes']
    # forward
    # rois = (128,5) im_info = (1,3) gt_vec = (2, 20), one-hot label for two streams
    net(im_data, rois, im_info, gt_vec)
    loss = net.loss
    train_loss += loss.data[0]
    step_cnt += 1

    # backward pass and update
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Log to screen
    if step % disp_interval == 0:
        duration = t.toc(average=False)
        fps = step_cnt / duration
        log_text = 'step %d, image: %s, loss: %.4f, fps: %.2f (%.2fs per batch), lr: %.9f, momen: %.4f, wt_dec: %.6f' % (
            step, blobs['im_name'], train_loss / step_cnt, fps, 1./fps, lr, momentum, weight_decay)
        log_print(log_text, color='green', attrs=['bold'])
        re_cnt = True

    #TODO: evaluate the model every 5000 iterations (N defined in handout)
    if step%100 == 0 and step>0:
        save_name = 'test_'+str(step)
        net.eval()
        imdb_test.competition_mode(on=True)
        ap = test_net(save_name, net, imdb_test,
                       cfg.TRAIN.BATCH_SIZE, thresh=thresh, visualize=use_visdom)
#       # import pdb; pdb.set_trace()
        for i in range(len(imdb.classes)):
            cls_name = imdb.classes[i]
            cur_ap = ap[i]
            logger.scalar_summary('{}_AP'.format(cls_name), cur_ap, step)
        viz.updateTrace(X=np.array([step]), Y=np.array([np.average(ap)]), win=ap_win,name='train_ap')




    #TODO: Perform all visualizations here
    #You can define other interval variable if you want (this is just an
    #example)
    #The intervals for different things are defined in the handout
#     if visualize and step%vis_interval==0:
#         #TODO: Create required visualizations
#         print("visualizing")
#         if use_tensorboard:
#             print('Logging to Tensorboard')
#             data_log.scalar_summary(tag='train/loss', value=loss, step=step)
            
#         if use_visdom:
#             print('Logging to visdom')            
#             viz.updateTrace(X=step, Y=loss.data[0], win=loss_win,name='train_loss')
    
    # plot the loss every 500 iterations
    if is_plot_loss and step%100==0:
        #TODO: Create required visualizations
#         import pdb; pdb.set_trace()
        if use_tensorboard:
            print('Tensorboard loss')
            data_log.scalar_summary(tag='train/loss', value=loss, step=step)
            
        if use_visdom:
            print('Visdom loss')        
            viz.updateTrace(X=np.array([step]), Y=np.array([loss.data[0]]), win=loss_win,name='train_loss')
    
    # plot the histogram of weights and histogram of gradients of weights in tensorboard every 2000 iterations
    if is_plot_weights and step%100==0:
        #TODO: Create required visualizations
        if use_tensorboard:
            print('Tensorboard weights')
            data_log.model_param_histo_summary(model=net, step=step)


    
    # Save model occasionally 
    if (step % cfg.TRAIN.SNAPSHOT_ITERS == 0) and step > 0:
        save_name = os.path.join(output_dir, '{}_{}.h5'.format(cfg.TRAIN.SNAPSHOT_PREFIX,step))
        network.save_net(save_name, net)
        print('Saved model to {}'.format(save_name))

    if step in lr_decay_steps:
        lr *= lr_decay
        optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    if re_cnt:
        tp, tf, fg, bg = 0., 0., 0, 0
        train_loss = 0
        step_cnt = 0
        t.tic()
        re_cnt = False



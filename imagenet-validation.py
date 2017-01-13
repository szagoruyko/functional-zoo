# dependencies
import re
import os
import argparse
import torch
from tqdm import tqdm
import cv2
import hickle as hkl
import numpy as np
import torch.utils.data
import torchnet as tnt
from torchvision import cvtransforms
import torchvision.datasets as datasets
import torch.nn.functional as F
from torch.autograd import Variable

# input arguments
parser = argparse.ArgumentParser(description = 'PyTorch ImageNet validation')
parser.add_argument('--imagenetpath', metavar='PATH', required=True,
                    help='path to dataset')
parser.add_argument('--numthreads', default=4, type=int, metavar='N',
                    help='number of data loading threads (default: 4)')
parser.add_argument('--model', metavar='PATH', required=True,
                    help='path to model')


def define_model(params):
    blocks = [sum([re.match('group%d.block\d+.conv0.weight'%j, k) is not None
                   for k in params.keys()]) for j in range(4)]

    def conv2d(input, params, base, stride=1, padding=0):
        return F.conv2d(input, params[base + '.weight'], params[base + '.bias'], stride, padding)

    def group(input, params, base, stride, n):
        o = input
        for i in range(0,n):
            b_base = ('%s.block%d.conv') % (base, i)
            x = o
            o = conv2d(x, params, b_base + '0', padding=1, stride=i==0 and stride or 1)
            o = F.relu(o)
            o = conv2d(o, params, b_base + '1', padding=1)
            if i == 0 and stride != 1:
                o += conv2d(x, params, b_base + '_dim', stride=stride)
            else:
                o += x
            o = F.relu(o)
        return o

    def f(input, params):
        o = F.conv2d(input, params['conv0.weight'], params['conv0.bias'], 2, 3)
        o = F.relu(o)
        o = F.max_pool2d(o, 3, 2, 1)
        o_g0 = group(o, params, 'group0', 1, blocks[0])
        o_g1 = group(o_g0, params, 'group1', 2, blocks[1])
        o_g2 = group(o_g1, params, 'group2', 2, blocks[2])
        o_g3 = group(o_g2, params, 'group3', 2, blocks[3])
        o = F.avg_pool2d(o_g3, 7)
        o = o.view(o.size(0), -1)
        o = F.linear(o, params['fc.weight'], params['fc.bias'])
        return o
    return f


def main():

    # parse input arguments
    args = parser.parse_args()

    def cvload(path):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    # set up data loader
    print("| setting up data loader...")
    valdir = os.path.join(args.imagenetpath, 'val')
    ds = datasets.ImageFolder(valdir, tnt.transform.compose([
            cvtransforms.Scale(256),
            cvtransforms.CenterCrop(224),
            lambda x: x.astype(np.float32) / 255.0,
            cvtransforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225]),
            lambda x: x.transpose(2,0,1).astype(np.float32),
            torch.from_numpy,
            ]), loader = cvload)
    train_loader = torch.utils.data.DataLoader(ds,
        batch_size=256, shuffle=False,
        num_workers=args.numthreads, pin_memory=False)

    params = hkl.load(args.model)
    params = {k: Variable(torch.from_numpy(v).cuda()) for k,v in params.iteritems()}

    f = define_model(params)

    class_err = tnt.meter.ClassErrorMeter(topk=[1,5], accuracy=True)

    for sample in tqdm(train_loader):
        inputs = Variable(sample[0].cuda(), volatile=True)
        targets = sample[1]
        class_err.add(f(inputs, params).data, targets)

    print 'Validation top1/top5 accuracy:'
    print class_err.value()

if __name__ == '__main__':
    main()

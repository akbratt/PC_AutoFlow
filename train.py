import torch
import numpy as np
import transforms
import time
import preprocess
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
import utils
import models
import sys
import argparse
import os

parser = argparse.ArgumentParser(description='Automated segmentation for'+\
        'PC-CMR -- training')
parser.add_argument('--datapath', type=str, default='.',
                    help='path to image data')
parser.add_argument('--cuda', type=bool, default=True,
                    help='whether to use cuda')
parser.add_argument('--batch_size', type=int, default=24,
                    help='batch size')
parser.add_argument('--log_interval', type=int, default=100,
                    help='interval between training status logs')
args = parser.parse_args()

def main():
    in_z = 0
    volpath = os.path.join(args.datapath, 'train')
    segpath = volpath


    batch_size = args.batch_size
    orig_dim = 256
    sqr = transforms.Square()
    aff = transforms.Affine()
    crop = transforms.RandomCrop(224)
    scale = transforms.Scale(orig_dim)
    rotate = transforms.Rotate(0.5, 30)
    noise = transforms.Noise(0.02)
    flip = transforms.Flip()
    transform_plan = [sqr, scale, aff, rotate, crop, flip, noise]
    lr = 1e-4
    series_names = ['Mag']
    seg_series_names = ['AV']


    model = models.Net23(2)

    model.cuda()
    optimizer = optim.RMSprop(model.parameters(), lr=lr)

    out_z, center_crop_sz = utils.get_out_size(orig_dim, in_z,\
            transform_plan, model)

    t0 = time.time()

    counter = 0
    print_interval = args.log_interval
    model.train()
    for i in range(200000000000000000000000000000000):
        weight = torch.FloatTensor([0.2,0.8]).cuda()
        vol, seg, inds = preprocess.get_batch(volpath, segpath, batch_size, in_z,\
                out_z, center_crop_sz, series_names, seg_series_names,\
                transform_plan, 8, nrrd=True)
        vol = torch.unsqueeze(vol, 1)
        vol = Variable(vol).cuda()
        seg = Variable(seg).cuda()

        out = model(vol).squeeze()

        loss = F.cross_entropy(out, seg, weight=weight)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        counter += 1

        sys.stdout.write('\r{:.2f}%'.format(counter*batch_size/print_interval))
        sys.stdout.flush()

        if counter*batch_size >= print_interval and i > 0:

            seg_hot = utils.get_hot(seg.data.cpu().numpy(), out.size()[-3])
            seg_hot = np.transpose(seg_hot, axes=[1,0,2,3])
            out_hot = np.argmax(out.data.cpu().numpy(), axis=1)
            out_hot = utils.get_hot(out_hot, out.size()[-3])
            out_hot = np.transpose(out_hot, axes=[1,0,2,3])

            dce2 = utils.dice(seg_hot[:,1:], out_hot[:,1:])

            vol_ind = utils.get_liver(seg)
            v = vol.data.cpu().numpy()[vol_ind].squeeze()
            real_seg = seg[vol_ind].data.cpu().numpy()
            out_plt = out.data.cpu().numpy()[vol_ind]
            out_plt = np.argmax(out_plt, axis=0)
            fake_seg = out_plt
            masked_real = utils.makeMask(v, real_seg, out.size()[-3], 0.5)
            masked_fake = utils.makeMask(v, fake_seg, out.size()[-3], 0.5)

            fig = plt.figure(1)
            fig.suptitle('Volume {} ; Dice = {:.2f}'.format(\
                    inds[vol_ind],dce2))
            v = fig.add_subplot(1,2,1)
            v.set_title('real')
            plt.imshow(masked_real)
            sreal = fig.add_subplot(1,2,2)
            sreal.set_title('fake')
            plt.imshow(masked_fake)
            outfile = os.path.join(args.datapath, 'out.png')
            plt.savefig(outfile, dpi=200)
            plt.clf()

            print(('\rIteration {}: Block completed in {:.2f} sec ; Loss = {:.2f}')\
                    .format(i*batch_size,time.time()-t0, loss.data.cpu()[0]))
            checkpoint_file = os.path.join(args.datapath,\
                    'model_checkpoint.pth')
            torch.save(model.state_dict(),checkpoint_file)
            counter = 0

            t0 = time.time()


if __name__ == '__main__':
    main()

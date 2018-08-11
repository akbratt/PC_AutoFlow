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
import densenet
import densenet2
import re
# from torch.nn import init

in_z = 0
fold = '0'
tag = 'res'
# volpath = '/home/alex/LiverSegmentation/5mm_train_3'
# segpath = '/home/alex/LiverSegmentation/5mm_train_3'
volpath = '/home/alex/samsung_512/CMR_PC/folds/fold_' + fold + '/train'
segpath = '/home/alex/samsung_512/CMR_PC/folds/fold_' + fold + '/train'
progress_file = '/home/alex/samsung_512/CMR_PC/folds/fold_' + fold +\
        '/' + tag + '.csv'


test_volpath = volpath.replace('train', 'test')
test_segpath = segpath.replace('train', 'test')
f = preprocess.gen_filepaths(test_segpath)

mult_inds = []
for i in f:
    if 'volume' in i:
        mult_inds.append(int(re.findall('\d+', i)[0]))

mult_inds = sorted(mult_inds)

mult_inds = np.unique(mult_inds)


batch_size = 24
sqr = transforms.Square()
pad = transforms.Pad(36)
aff = transforms.Affine()
crop = transforms.RandomCrop(224)
scale = transforms.Scale(256)
rotate = transforms.Rotate(0.5, 30)
noise = transforms.Noise(0.02)
flip = transforms.Flip()
orig_dim = 256
transform_plan = [sqr, scale, aff, rotate, crop, flip, noise]
lr = 1e-4
initial_depth = in_z*2+1
series_names = ['Mag']
seg_series_names = ['AV']


center = transforms.CenterCrop2(224)
t_transform_plan = [sqr, scale, center]
volpaths, segpaths = utils.get_paths(mult_inds, f, series_names, \
        seg_series_names, test_volpath, test_segpath)


#todo
#add different type of scaling
#add depth padding
if tag == 'res':
    model = models.Net23(2)
elif tag == 'dense_pre':
    model = densenet2.densenet121(True)
elif tag == 'dense_nopre':
    model = densenet2.densenet121(False)
elif tag == 'dense_test':
    model = densenet.densenet_small()
else:
    raise ValueError('Invalid tag')

# model = densenet.densenet121()
model.cuda()
# model.load_state_dict(torch.load(\
        # '/home/alex/samsung_512/CMR_PC/folds/fold_0/dense_pre2.pth'))
# model.load_state_dict(torch.load(\
        # '/home/alex/samsung_512/CMR_PC/checkpoint_dense.pth'))

optimizer = optim.RMSprop(model.parameters(), lr=lr)
# optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
# optimizer = optim.Adam(model.parameters(), lr=lr)

out_z, center_crop_sz = utils.get_out_size(orig_dim, in_z,\
        transform_plan, model)
center = transforms.CenterCrop(center_crop_sz)
depth_center = transforms.DepthCenterCrop(out_z)

# t_transform_plan = []
# t_transform_plan = transform_plan
# t_out_z, t_center_crop_sz = utils.get_out_size(orig_dim, in_z,\
        # t_transform_plan, model)
# t_center = transforms.CenterCrop(t_center_crop_sz)
# t_depth_center = transforms.DepthCenterCrop(t_out_z)

t0 = time.time()

# weight = torch.FloatTensor(preprocess.get_weights(volpath, segpath,
    # 8, clip_val=1, nrrd=False)).cuda()

# print(weight)
# print('generating weights took {:.2f} seconds'.format(time.time()-t0))
counter = 0
print_interval = 490
model.train()
progress = [[0,0,0,0,0]]
for i in range(200000000000000000000000000000000):
    # model.train()
    weight = torch.FloatTensor([0.2,0.8]).cuda()
    # weight = torch.FloatTensor([0.0012, 0.0525, 0.9463]).cuda()
    vol, seg, inds = preprocess.get_batch(volpath, segpath, batch_size, in_z,\
            out_z, center_crop_sz, series_names, seg_series_names,\
            transform_plan, 8, nrrd=True)
    vol = torch.unsqueeze(vol, 1)
    vol = Variable(vol).cuda()
    seg = Variable(seg).cuda()

    out = model(vol).squeeze()

    # loss0 = F.cross_entropy(out0, seg0, weight=weight0)
    # loss1 = F.cross_entropy(out1, seg, weight=weight1)

    # loss = loss0
    # if i * batch_size > 100:
    # loss = loss0 + loss1
    loss = F.cross_entropy(out, seg, weight=weight)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    counter += 1

    sys.stdout.write('\r{:.2f}%'.format(counter*batch_size/print_interval))
    sys.stdout.flush()

    if counter*batch_size >= print_interval and i > 0:
        # this is a hack to increase performance
        # if np.random.rand() < 0.2:
            # optimizer = optim.RMSprop(model.parameters(), lr=lr)

        counter = 0
        dce, jc, hd, assd = utils.test_net_cheap('', '',\
                mult_inds, in_z, model,\
                t_transform_plan, orig_dim, 5, '', 2,\
                2, volpaths, segpaths, nrrd=True,\
                vol_only=False, get_dice=True, verbose=False)

        seg_hot = utils.get_hot(seg.data.cpu().numpy(), out.size()[-3])
        seg_hot = np.transpose(seg_hot, axes=[1,0,2,3])
        out_hot = np.argmax(out.data.cpu().numpy(), axis=1)
        out_hot = utils.get_hot(out_hot, out.size()[-3])
        out_hot = np.transpose(out_hot, axes=[1,0,2,3])

        dce2 = utils.dice(seg_hot[:,1:], out_hot[:,1:])
        # hd2, assd2 = utils.get_dists_volumetric(seg.data.cpu().numpy().astype(np.int64),\
                # np.argmax(out.data.cpu().numpy(), axis=1))

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
        v = fig.add_subplot(2,2,1)
        v.set_title('real')
        plt.imshow(masked_real)
        sreal = fig.add_subplot(2,2,2)
        sreal.set_title('fake')
        plt.imshow(masked_fake)
        test_real = fig.add_subplot(2,2,3)
        test_real.set_title('test_real')
        plt.imshow(masked_real)
        test = fig.add_subplot(2,2,4)
        test.set_title('test fake')
        fig.tight_layout()
        plt.imshow(masked_fake)
        plt.savefig('/home/alex/samsung_512/CMR_PC/folds/fold_' + fold +\
                '/out.png', dpi=200)
        plt.clf()

        progress.append([i*batch_size, np.mean(dce), np.mean(jc), np.mean(hd),\
                np.mean(assd)])
        prg = np.array(progress)
        np.savetxt(progress_file, np.array(prg))
        plt.plot(prg[:,0], prg[:,1])
        plt.tight_layout()
        plt.savefig('/home/alex/samsung_512/CMR_PC/folds/fold_' + fold +\
                '/' + tag + '_plt.png')
        plt.clf()

        print(('\rStep {} completed in {:.2f} sec ; Loss = {:.2f}' +\
                '; Dice = {:.3f} ; JC = {:.2f} ; HD = {:.2f} ; ASSD = {:.2f}')\
                .format(i*batch_size,time.time()-t0, loss.data.cpu()[0],\
                np.mean(dce),np.mean(jc),np.mean(hd),np.mean(assd)))
        if np.mean(dce) > 0.93:
            torch.save(model.state_dict(),\
                    '/home/alex/samsung_512/CMR_PC/folds/fold_' + fold +\
                    '/' + tag + '_' + '{:.3f}'.format(np.mean(dce)) + 'testing.pth')

        t0 = time.time()
        # print(weight)

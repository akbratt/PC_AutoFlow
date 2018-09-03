import torch
import numpy as np
import transforms
import preprocess
import utils
import models
import re
import pandas as pd

in_z = 0
cuda = False

test_volpath ='/home/alex/samsung_512/CMR_PC/external_data_nathan_definitive/test'
test_segpath = test_volpath
out_file = '/home/alex/samsung_512/CMR_PC/external_data_nathan_definitive/fakes'
double_vol = False
model = models.Net23(2)
if cuda:
    model.cuda()

model.load_state_dict(torch.load(\
        '/home/alex/samsung_512/CMR_PC/external_data_nathan_definitive/checkpoint.pth'))

batch_size = 5
t_batch_size = 1
pad = transforms.Pad(2)
sqr = transforms.Square()
center = transforms.CenterCrop2(224)
crop = transforms.RandomCrop(256)
scale = transforms.Scale(256)
orig_dim = 256
transform_plan = [sqr, scale, center]
num_labels=2
num_labels_final = 2
series_names = ['Mag']
seg_series_names = ['AV']

f = preprocess.gen_filepaths(test_segpath)

mult_inds = []
for i in f:
    if 'volume' in i:
        mult_inds.append(int(re.findall('\d+', i)[0]))

mult_inds = sorted(mult_inds)

mult_inds = np.unique(mult_inds)

volpaths, segpaths = utils.get_paths(mult_inds, f, series_names, \
        seg_series_names, test_volpath, test_segpath)

t_transform_plan = transform_plan

out = utils.test_net_cheap(test_volpath, test_segpath, mult_inds, in_z, model,\
        t_transform_plan, orig_dim, batch_size, out_file, num_labels,\
        num_labels_final, volpaths, segpaths, nrrd=True,\
        vol_only=double_vol, get_dice=True, make_niis=False, cuda=cuda)
out.to_csv('/home/alex/samsung_512/CMR_PC/external_data_nathan_definitive/out.csv',\
        index=False)

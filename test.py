import torch
import numpy as np
import transforms
import preprocess
import utils
import models
import re
import argparse
import os


parser = argparse.ArgumentParser(description='Automated segmentation for'+\
        'PC-CMR -- testing')
parser.add_argument('--datapath', type=str, default='.',
                    help='path to image data')
parser.add_argument('--cuda', type=bool, default=True,
                    help='whether to use cuda')
parser.add_argument('--batch_size', type=int, default=5,
                    help='batch size')
args = parser.parse_args()

def main():
    in_z = 0

    test_volpath = os.path.join(args.datapath, 'test')
    out_file = os.path.join(args.datapath, 'fakes')
    checkpoint_path = os.path.join(args.datapath, 'checkpoint.pth')
    test_segpath = test_volpath
    double_vol = False
    model = models.Net23(2)
    cuda = args.cuda
    if cuda:
        model.cuda()

    model.load_state_dict(torch.load(checkpoint_path))

    batch_size = args.batch_size
    orig_dim = 256
    sqr = transforms.Square()
    center = transforms.CenterCrop2(224)
    scale = transforms.Scale(orig_dim)
    transform_plan = [sqr, scale, center]
    num_labels=2
    series_names = ['Mag']
    seg_series_names = ['AV']

    f = preprocess.gen_filepaths(test_segpath)

    mult_inds = []
    for i in f:
        if 'volume' in i:
            mult_inds.append(int(re.findall('\d+', i)[0]))

    mult_inds = sorted(mult_inds)

    mult_inds = np.unique(mult_inds)
    mult_inds = mult_inds[0:5]

    volpaths, segpaths = utils.get_paths(mult_inds, f, series_names, \
            seg_series_names, test_volpath, test_segpath)

    out = utils.test_net_cheap(mult_inds, in_z, model,\
            transform_plan, orig_dim, batch_size, out_file, num_labels,\
            volpaths, segpaths, nrrd=True, vol_only=double_vol,\
            get_dice=True, make_niis=False, cuda=cuda)
    out_csv = os.path.join(args.datapath, 'out.csv')
    out.to_csv(out_csv, index=False)

if __name__ == '__main__':
    main()

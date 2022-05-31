import os
import argparse
import pickle
import numpy as np

import torch
import torch.nn.functional as F

import dnnlib
import legacy

from util.utilgan import basename, calc_init_res
try: # progress bar for notebooks 
    get_ipython().__class__.__name__
    from util.progress_bar import ProgressIPy as ProgressBar
except: # normal console
    from util.progress_bar import ProgressBar

parser = argparse.ArgumentParser()
parser.add_argument('--source', required=True, help='Source model path')
parser.add_argument('--out_dir', default='./', help='Output directory for reduced/reconstructed model')
parser.add_argument('-r', '--reconstruct', action='store_true', help='Reconstruct model (add internal arguments)')
parser.add_argument('-s', '--res', default=None, help='Target resolution in format X-Y')
parser.add_argument('-a', '--alpha', action='store_true', help='Add alpha channel for RGBA processing')
parser.add_argument('-l', '--labels', default=0, type=int, help='Make conditional model')
parser.add_argument('-f', '--full', action='store_true', help='Save full model')
parser.add_argument('-v', '--verbose', action='store_true')
a = parser.parse_args()

if a.res is not None: 
    a.res = [int(s) for s in a.res.split('-')][::-1]
    if len(a.res) == 1: a.res = a.res + a.res

def load_pkl(filepath):
    with dnnlib.util.open_url(filepath) as f:
        nets = legacy.load_network_pkl(f, custom=False) # ['G', 'D', 'G_ema', 'training_set_kwargs', 'augment_pipe']
    return nets
    
def save_pkl(nets, filepath):
    with open(filepath, 'wb') as file:
        pickle.dump(nets, file) # , protocol=pickle.HIGHEST_PROTOCOL

def create_model(net_in, data_shape, labels=None, full=False, custom=False, init=False):
    init_res, resolution, res_log2 = calc_init_res(data_shape[1:])
    net_in['G_ema'].img_resolution = resolution
    net_in['G_ema'].img_channels = data_shape[0]
    net_in['G_ema'].init_res = init_res
    net_out = legacy.create_networks(net_in, full=full, custom=custom, init=init, labels=labels)
    return net_out

def add_channel(x, subnet): # [BCHW]
    if subnet == 'D': # pad second dim [1]
        padding = [0] * (len(x.shape)-2)*2
        padding += [0,1,0,0]
    else: # pad last dim [-1]
        padding = [0] * (len(x.shape)-1)*2
        padding += [0,1]
    y = F.pad(x, padding, 'constant', 1)
    return y

def pad_up_to(x, size, type='side'):
    sh = x.shape
    if list(x.shape) == list(size): return x
    padding = []
    for i, s in enumerate(size):
        p0 = (s-sh[i]) // 2
        p1 = s-sh[i] - p0
        padding = padding + [p0,p1]
    y = F.pad(x, padding[::-1], 'constant', 0)
    return y

def copy_vars(src_net, tgt_net, add_alpha=False, xtile=False) -> None:
    for subnet in ['G_ema', 'G', 'D']:
        if subnet in src_net.keys() and subnet in tgt_net.keys():
            src_dict = src_net[subnet].state_dict()
            tgt_dict = tgt_net[subnet].state_dict()
            vars = [name for name in src_dict.keys() if name in tgt_dict.keys()]
            pbar = ProgressBar(len(vars))
            for name in vars:
                source_shape = src_dict[name].shape
                target_shape = tgt_dict[name].shape
                if source_shape == target_shape:
                    tgt_dict[name].copy_(src_dict[name]).requires_grad_(False)
                else:
                    if add_alpha:
                        update = add_channel(src_dict[name], subnet)
                        assert target_shape == update.shape, 'Diff shapes yet: src %s tgt %s' % (str(update.shape), str(target_shape))
                        tgt_dict[name].copy_(update).requires_grad_(False)
                    elif xtile:
                        assert len(source_shape) == len(target_shape), "Diff shape ranks: src %s tgt %s" % (str(source_shape), str(target_shape))
                        try:
                            update = src_dict[name][:target_shape[0], :target_shape[1], ...] # !!! corrects only first two dims
                        except:
                            update = src_dict[name][:target_shape[0]]
                        if np.greater(target_shape, source_shape).any():
                            tile_count = [target_shape[i] // source_shape[i] for i in range(len(source_shape))]
                            update = src_dict[name].repeat(*tile_count) # [512,512] => [1024,512]
                        if a.verbose is True: print(name, tile_count, source_shape, '=>', target_shape, '\n\n') # G_mapping/Dense0, D/Output
                        tgt_dict[name].copy_(update).requires_grad_(False)
                    else: # crop/pad
                        update = pad_up_to(src_dict[name], target_shape)
                        if a.verbose is True: print(name, source_shape, '=>', update.shape, '\n\n')
                        tgt_dict[name].copy_(update).requires_grad_(False)
                pbar.upd(name)

def main():

    net_in = load_pkl(a.source)
    Gs_in = net_in['G_ema']
    if hasattr(Gs_in, 'output_shape'):
        out_shape = Gs_in.output_shape
        print(' Loading model', a.source, out_shape)
        _, res_in, _  = calc_init_res(out_shape[1:])
    else: # original model
        res_in = Gs_in.img_resolution
        out_shape = [None, Gs_in.img_channels, res_in, res_in]
    # netdict = net_in['G_ema'].state_dict()
    # for k in netdict.keys(): 
        # print(k, netdict[k].shape)
    
    if a.res is not None or a.alpha is True:
        if a.res is None: a.res = out_shape[2:]
        colors = 4 if a.alpha is True else out_shape[1]
        _, res_out, _ = calc_init_res([colors, *a.res])

        if res_in != res_out or a.alpha is True: # add or remove layers
            assert 'G' in net_in.keys() and 'D' in net_in.keys(), " !! G/D subnets not found in source model !!"
            data_shape = [colors, res_out, res_out]
            print(' Reconstructing full model with shape', data_shape)
            net_out = create_model(net_in, data_shape, full=True)
            copy_vars(net_in, net_out, add_alpha=True)
            a.full = True

        if a.res[0] != res_out or a.res[1] != res_out: # crop or pad layers
            data_shape = [colors, *a.res]
            net_out = create_model(net_in, data_shape, full=True)
            copy_vars(net_in, net_out)

    if a.labels is not None:
        assert 'G' in net_in.keys() and 'D' in net_in.keys(), " !! G/D subnets not found in source model !!"
        print(' Reconstructing full model with labels', a.labels)
        data_shape = out_shape[1:]
        net_out = create_model(net_in, data_shape, labels=a.labels, full=True)
        copy_vars(net_in, net_out, xtile=True)
        a.full = True

    if a.labels is None and a.res is None and a.alpha is not True:
        if a.reconstruct is True:
            print(' Reconstructing model with same size /', 'full' if a.full else 'Gs')
            data_shape = out_shape[1:]
            net_out = create_model(net_in, data_shape, full=a.full, init=True)
        else:
            net_out = dict(G_ema = Gs_in)

    out_name = basename(a.source)
    if a.res is not None:    out_name += '-%dx%d' % (a.res[1], a.res[0])
    if a.alpha is True:      out_name += 'a'
    if a.labels is not None: out_name += '-c%d' % a.labels
    if not a.full:           out_name += '-Gs'

    save_pkl(net_out, os.path.join(a.out_dir, '%s.pkl' % out_name))
    print(' Done')


if __name__ == '__main__':
    main()

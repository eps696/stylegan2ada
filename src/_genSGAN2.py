import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import os.path as osp
import argparse
import numpy as np
from imageio import imsave

import torch

import dnnlib
import legacy

from util.utilgan import latent_anima, basename, img_read
try: # progress bar for notebooks 
    get_ipython().__class__.__name__
    from util.progress_bar import ProgressIPy as ProgressBar
except: # normal console
    from util.progress_bar import ProgressBar

desc = "Customized StyleGAN2-ada on PyTorch"
parser = argparse.ArgumentParser(description=desc)
parser.add_argument('-o', '--out_dir',  default='_out', help='output directory')
parser.add_argument('-m', '--model',    default='models/ffhq-1024.pkl', help='path to pkl checkpoint file')
parser.add_argument('-l', '--labels',   default=None, type=int, help='labels/categories for conditioning')
# custom
parser.add_argument('-s', '--size',     default=None, help='output resolution, set in X-Y format')
parser.add_argument('-sc', '--scale_type', default='pad', help="main types: pad, padside, symm, symmside")
parser.add_argument('-lm', '--latmask', default=None, help='external mask file (or directory) for multi latent blending')
parser.add_argument('-n', '--nXY',      default='1-1', help='multi latent frame split count by X (width) and Y (height)')
parser.add_argument('--splitfine', type=float, default=0, help='multi latent frame split edge sharpness (0 = smooth, higher => finer)')
parser.add_argument('-tr', '--trunc',   default=0.8, type=float, help='truncation psi 0..1 (lower = stable, higher = various)')
parser.add_argument('-d', '--digress',  default=0, type=float, help='distortion technique by Aydao (strength of the effect)') 
parser.add_argument('--save_lat', action='store_true', help='save latent vectors to file')
parser.add_argument('--seed',           default=None, type=int)
parser.add_argument('-v', '--verbose',  action='store_true')
# animation
parser.add_argument('--frames', default='200-25', help='total frames to generate, length of interpolation step')
parser.add_argument("--cubic", action='store_true', help="use cubic splines for smoothing")
parser.add_argument("--gauss", action='store_true', help="use Gaussian smoothing")
a = parser.parse_args()

if a.size is not None: 
    a.size = [int(s) for s in a.size.split('-')][::-1]
    if len(a.size) == 1: a.size = a.size * 2
[a.frames, a.fstep] = [int(s) for s in a.frames.split('-')]

def generate():
    os.makedirs(a.out_dir, exist_ok=True)
    if a.seed==0: a.seed = None
    np.random.seed(seed=a.seed)
    device = torch.device('cuda')

    # setup generator
    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.verbose = a.verbose
    Gs_kwargs.size = a.size
    Gs_kwargs.scale_type = a.scale_type
    
    # mask/blend latents with external latmask or by splitting the frame
    if a.latmask is None:
        nHW = [int(s) for s in a.nXY.split('-')][::-1]
        assert len(nHW)==2, ' Wrong count nXY: %d (must be 2)' % len(nHW)
        n_mult = nHW[0] * nHW[1]
        Gs_kwargs.countHW = nHW
        Gs_kwargs.splitfine = a.splitfine
        if a.verbose is True and n_mult > 1: print(' Latent blending w/split frame %d x %d' % (nHW[1], nHW[0]))
        lmask = [None]
    else:
        n_mult = 2
        if osp.isfile(a.latmask): # single file
            lmask = np.asarray([[img_read(a.latmask)[:,:,0] / 255.]]) # [1,1,h,w]
        elif osp.isdir(a.latmask): # directory with frame sequence
            lmask = np.expand_dims(np.asarray([img_read(f)[:,:,0] / 255. for f in img_list(a.latmask)]), 1) # [n,1,h,w]
        else:
            print(' !! Blending mask not found:', a.latmask); exit(1)
        if a.verbose is True: print(' Latent blending with mask', a.latmask, lmask.shape)
        lmask = np.concatenate((lmask, 1 - lmask), 1) # [frm,2,h,w]
        lmask = torch.from_numpy(lmask).to(device)
    
    # load base or custom network
    pkl_name = osp.splitext(a.model)[0]
    if '.pkl' in a.model.lower():
        custom = False
        print(' .. Gs from pkl ..', basename(a.model))
    else:
        custom = True
        print(' .. Gs custom ..', basename(a.model))
    with dnnlib.util.open_url(pkl_name + '.pkl') as f:
        Gs = legacy.load_network_pkl(f, custom=custom, **Gs_kwargs)['G_ema'].to(device) # type: ignore

    if a.verbose is True: print(' out shape', Gs.output_shape[1:])

    if a.verbose is True: print(' making timeline..')
    lats = [] # list of [frm,1,512]
    for i in range(n_mult):
        lat_tmp = latent_anima((1, Gs.z_dim), a.frames, a.fstep, cubic=a.cubic, gauss=a.gauss, seed=a.seed, verbose=False) # [frm,1,512]
        lats.append(lat_tmp) # list of [frm,1,512]
    latents = np.concatenate(lats, 1) # [frm,X,512]
    print(' latents', latents.shape)
    latents = torch.from_numpy(latents).to(device)
    frame_count = latents.shape[0]
    
    # distort image by tweaking initial const layer
    if a.digress > 0:
        try: init_res = Gs.init_res
        except: init_res = (4,4) # default initial layer size 
        dconst = []
        for i in range(n_mult):
            dc_tmp = a.digress * latent_anima([1, Gs.z_dim, *init_res], a.frames, a.fstep, cubic=True, seed=a.seed, verbose=False)
            dconst.append(dc_tmp)
        dconst = np.concatenate(dconst, 1)
    else:
        dconst = np.zeros([frame_count, 1, 1, 1, 1])
    dconst = torch.from_numpy(dconst).to(device)

    # labels / conditions
    label_size = Gs.c_dim
    if label_size > 0:
        labels = torch.zeros((frame_count, n_mult, label_size), device=device) # [frm,X,lbl]
        if a.labels is None:
            label_ids = []
            for i in range(n_mult):
                label_ids.append(random.randint(0, label_size-1))
        else:
            label_ids = [int(x) for x in a.labels.split('-')]
            label_ids = label_ids[:n_mult] # ensure we have enough labels
        for i, l in enumerate(label_ids):
            labels[:,i,l] = 1
    else:
        labels = [None]

    # warm up
    if custom:
        _ = Gs(latents[0], labels[0], lmask[0], dconst[0], noise_mode='const')
    else:
        _ = Gs(latents[0], labels[0], noise_mode='const')
    
    # generate images from latent timeline
    pbar = ProgressBar(frame_count)
    for i in range(frame_count):
    
        latent  = latents[i] # [X,512]
        label   = labels[i % len(labels)]
        latmask = lmask[i % len(lmask)] if lmask is not None else [None] # [X,h,w]
        dc      = dconst[i % len(dconst)] # [X,512,4,4]

        # generate multi-latent result
        if custom:
            output = Gs(latent, label, latmask, dc, truncation_psi=a.trunc, noise_mode='const')
        else:
            output = Gs(latent, label, truncation_psi=a.trunc, noise_mode='const')
        output = (output.permute(0,2,3,1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()

        # save image
        ext = 'png' if output.shape[3]==4 else 'jpg'
        filename = osp.join(a.out_dir, "%06d.%s" % (i,ext))
        imsave(filename, output[0])
        pbar.upd()

    # convert latents to dlatents, save them
    if a.save_lat is True:
        latents = latents.squeeze(1) # [frm,512]
        dlatents = Gs.mapping(latents, label) # [frm,18,512]
        if a.size is None: a.size = ['']*2
        filename = '{}-{}-{}.npy'.format(basename(a.model), a.size[1], a.size[0])
        filename = osp.join(osp.dirname(a.out_dir), filename)
        dlatents = dlatents.cpu().numpy()
        np.save(filename, dlatents)
        print('saved dlatents', dlatents.shape, 'to', filename)


if __name__ == '__main__':
    generate()

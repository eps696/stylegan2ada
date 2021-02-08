import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import os.path as osp
import argparse
import numpy as np
from imageio import imsave
import pickle
import cv2

import torch

import dnnlib
import legacy

from util.utilgan import load_latents, file_list, basename
try: # progress bar for notebooks 
    get_ipython().__class__.__name__
    from util.progress_bar import ProgressIPy as ProgressBar
except: # normal console
    from util.progress_bar import ProgressBar

desc = "Customized StyleGAN2 on Tensorflow"
parser = argparse.ArgumentParser(description=desc)
parser.add_argument('--vector_dir', default=None, help='Saved latent directions in *.npy format')
parser.add_argument('--base_lat', default=None, help='Saved latent vector as *.npy file')
parser.add_argument('--out_dir', default='_out/ttt', help='Output directory')
parser.add_argument('--model', default='models/ffhq-1024.pkl', help='path to checkpoint file')
parser.add_argument('--size', default=None, help='output resolution, set in X-Y format')
parser.add_argument('--scale_type', default='pad', help="may include pad, side, symm (try padside or sidesymm, etc.)")
parser.add_argument('--trunc', type=float, default=0.8, help='truncation psi 0..1 (lower = stable, higher = various)')
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--ops', default='cuda', help='custom op implementation (cuda or ref)')
# animation
parser.add_argument("--fstep", type=int, default=25, help="Number of frames for interpolation step")
a = parser.parse_args()

if a.size is not None: a.size = [int(s) for s in a.size.split('-')][::-1]

def generate_image(latent):
    args = [None, 0] if custom else [] # custom model?
    if use_d:
        img = Gs.synthesis(latent, *args, noise_mode='const')
    else:
        img = Gs(latent, None, *args, truncation_psi=a.trunc, noise_mode='const')
    img = (img.permute(0,2,3,1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()[0]
    return img

def render_latent_dir(latent, direction, coeff):
    new_latent = latent + coeff*direction 
    img = generate_image(new_latent)
    return img

def render_latent_mix(latent1, latent2, coeff):
    new_latent = latent1 * (1-coeff) + latent2 * coeff
    img = generate_image(new_latent)
    return img

def pingpong(x, delta):
    x = (x + delta) % 2
    if x > 1:
        x = 1 - (x%1)
        delta = -delta
    return x, delta

def get_coeffs_dir(lrange, count):
    dx = 1 / count
    x = -lrange[0] / (lrange[1] - lrange[0])
    xs = [0]
    for _ in range(count*2):
        x, dx = pingpong(x, dx)
        xs.append( x * (lrange[1] - lrange[0]) + lrange[0] )
    return xs
    
def make_loop(base_latent, direction, lrange, fcount, start_frame=0):
    coeffs = get_coeffs_dir(lrange, fcount//2)
    # pbar = ProgressBar(fcount)
    for i in range(fcount):
        img = render_latent_dir(base_latent, direction, coeffs[i])
        fname1 = os.path.join(a.out_dir, "%06d.jpg" % (i+start_frame))
        if i%2==0 and a.verbose is True:
            cv2.imshow('latent', img[:,:,::-1])
            cv2.waitKey(10)
        imsave(fname1, img)
        # pbar.upd()

def make_transit(lat1, lat2, fcount, start_frame=0):
    # pbar = ProgressBar(fcount)
    for i in range(fcount):
        img = render_latent_mix(lat1, lat2, i/fcount)
        fname = os.path.join(a.out_dir, "%06d.jpg" % (i+start_frame))
        if i%2==0 and a.verbose is True:
            cv2.imshow('latent', img[:,:,::-1])
            cv2.waitKey(10)
        imsave(fname, img)
        # pbar.upd()
        
def main():
    if a.vector_dir is not None:
        if a.vector_dir.endswith('/') or a.vector_dir.endswith('\\'): a.vector_dir = a.vector_dir[:-1]
    os.makedirs(a.out_dir, exist_ok=True)
    device = torch.device('cuda')
        
    global Gs, use_d, custom
        
    # setup generator
    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.verbose = a.verbose
    Gs_kwargs.size = a.size
    Gs_kwargs.scale_type = a.scale_type
    
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

    # load directions
    if a.vector_dir is not None:
        directions = []
        vector_list = file_list(a.vector_dir, 'npy')
        for v in vector_list: 
            direction = load_latents(v)
            if len(direction.shape) == 2: direction = np.expand_dims(direction, 0)
            directions.append(direction)
        directions = np.concatenate(directions)[:, np.newaxis] # [frm,1,18,512]
    else:
        print(' No vectors found'); exit()

    if len(direction[0].shape) > 1 and direction[0].shape[0] > 1: 
        use_d = True
    print(' directions', directions.shape, 'using d' if use_d else 'using w')
    directions = torch.from_numpy(directions).to(device)
    
    # latent direction range 
    lrange = [-0.5, 0.5]

    # load saved latents
    if a.base_lat is not None:
        base_latent = load_latents(a.base_lat)
        base_latent = torch.from_numpy(base_latent).to(device)
    else:
        print(' No NPY input given, making random')
        base_latent = np.random.randn(1, Gs.z_dim)
        if use_d:
            base_latent = Gs.mapping(base_latent, None) # [frm,18,512]

    pbar = ProgressBar(len(directions))
    for i, direction in enumerate(directions):
        make_loop(base_latent, direction, lrange, a.fstep*2, a.fstep*2 * i)
        pbar.upd()
        
        # make_transit(base_lats[i], base_lats[(i+1)%len(base_lats)], n, 2*n*i + n)


if __name__ == '__main__':
    main()


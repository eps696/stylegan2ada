import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import os.path as osp
import argparse
import numpy as np
from imageio import imsave

import torch

import dnnlib
import legacy

from util.utilgan import latent_anima, load_latents, file_list, basename
try: # progress bar for notebooks 
    get_ipython().__class__.__name__
    from util.progress_bar import ProgressIPy as ProgressBar
except: # normal console
    from util.progress_bar import ProgressBar

desc = "Customized StyleGAN2 on Tensorflow"
parser = argparse.ArgumentParser(description=desc)
parser.add_argument('--dlatents', default=None, help='Saved dlatent vectors in single *.npy file or directory with such files')
parser.add_argument('--style_dlat', default=None, help='Saved latent vector for hi res (style) features')
parser.add_argument('--out_dir', default='_out', help='Output directory')
parser.add_argument('--model', default='models/ffhq-1024-f.pkl', help='path to checkpoint file')
parser.add_argument('--size', default=None, help='Output resolution')
parser.add_argument('--scale_type', default='pad', help="main types: pad, padside, symm, symmside")
parser.add_argument('--trunc', type=float, default=1, help='Truncation psi 0..1 (lower = stable, higher = various)')
parser.add_argument('--digress', type=float, default=0, help='distortion technique by Aydao (strength of the effect)') 
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--ops', default='cuda', help='custom op implementation (cuda or ref)')
# animation
parser.add_argument("--fstep", type=int, default=25, help="Number of frames for smooth interpolation")
parser.add_argument("--cubic", action='store_true', help="Use cubic splines for smoothing")
a = parser.parse_args()

if a.size is not None: a.size = [int(s) for s in a.size.split('-')][::-1]

def main():
    os.makedirs(a.out_dir, exist_ok=True)
    device = torch.device('cuda')

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

    dlat_shape = (1, Gs.num_ws, Gs.w_dim) # [1,18,512]
    
    # read saved latents
    if a.dlatents is not None and osp.isfile(a.dlatents):
        key_dlatents = load_latents(a.dlatents)
        if len(key_dlatents.shape) == 2: key_dlatents = np.expand_dims(key_dlatents, 0)
    elif a.dlatents is not None and osp.isdir(a.dlatents):
        # if a.dlatents.endswith('/') or a.dlatents.endswith('\\'): a.dlatents = a.dlatents[:-1]
        key_dlatents = []
        npy_list = file_list(a.dlatents, 'npy')
        for npy in npy_list: 
            key_dlatent = load_latents(npy)
            if len(key_dlatent.shape) == 2: key_dlatent = np.expand_dims(key_dlatent, 0)
            key_dlatents.append(key_dlatent)
        key_dlatents = np.concatenate(key_dlatents) # [frm,18,512]
    else:
        print(' No input dlatents found'); exit()
    key_dlatents = key_dlatents[:, np.newaxis] # [frm,1,18,512]
    print(' key dlatents', key_dlatents.shape)
    
    # replace higher layers with single (style) latent
    if a.style_dlat is not None:
        print(' styling with dlatent', a.style_dlat)
        style_dlatent = load_latents(a.style_dlat)
        while len(style_dlatent.shape) < 4: style_dlatent = np.expand_dims(style_dlatent, 0)
        # try replacing 5 by other value, less than Gs.num_ws
        key_dlatents[:, :, range(5, Gs.num_ws), :] = style_dlatent[:, :, range(5, Gs.num_ws), :]
       
    frames = key_dlatents.shape[0] * a.fstep
    
    dlatents = latent_anima(dlat_shape, frames, a.fstep, key_latents=key_dlatents, cubic=a.cubic, verbose=True) # [frm,1,512]
    print(' dlatents', dlatents.shape)
    frame_count = dlatents.shape[0]
    dlatents = torch.from_numpy(dlatents).to(device)

    # distort image by tweaking initial const layer
    if a.digress > 0:
        try: init_res = Gs.init_res
        except: init_res = (4,4) # default initial layer size 
        dconst = a.digress * latent_anima([1, Gs.z_dim, *init_res], frame_count, a.fstep, cubic=True, verbose=False)
    else:
        dconst = np.zeros([frame_count, 1, 1, 1, 1])
    dconst = torch.from_numpy(dconst).to(device)
    
    # generate images from latent timeline
    pbar = ProgressBar(frame_count)
    for i in range(frame_count):
    
        # generate multi-latent result
        if custom:
            output = Gs.synthesis(dlatents[i], None, dconst[i], noise_mode='const')
        else:
            output = Gs.synthesis(dlatents[i], noise_mode='const')
        output = (output.permute(0,2,3,1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()

        ext = 'png' if output.shape[3]==4 else 'jpg'
        filename = osp.join(a.out_dir, "%06d.%s" % (i,ext))
        imsave(filename, output[0])
        pbar.upd()

        
if __name__ == '__main__':
    main()


import os
import sys
import math
import argparse
import numpy as np
import pickle

import torch

import dnnlib, legacy
try: # progress bar for notebooks 
    get_ipython().__class__.__name__
    from util.progress_bar import ProgressIPy as ProgressBar
except: # normal console
    from util.progress_bar import ProgressBar

def get_args():
    parser = argparse.ArgumentParser(description="Rosinality (pytorch) to Nvidia (pkl) checkpoint converter")
    parser.add_argument("--model_pkl", metavar="PATH", help="path to the source sg2ada pytorch (pkl) weights")
    parser.add_argument("--model_pt", metavar="PATH", help="path to the updated pytorch (pt) weights")
    parser.add_argument("--channel_multiplier", type=int, default=2, help="channel multiplier factor. config-f = 2, else = 1")
    args = parser.parse_args()
    return args

def load_pkl(filepath):
    with dnnlib.util.open_url(filepath) as f:
        nets = legacy.load_network_pkl(f, custom=False)
    return nets
    
def save_pkl(nets, filepath):
    with open(filepath, 'wb') as file:
        pickle.dump(nets, file)

def update(tgt_dict, name, value):
    tgt_dict[name].copy_(value).requires_grad_(False)

def convert_modconv(tgt_dict, src_dict, target_name, source_name):
    conv_weight = src_dict[source_name + ".conv.weight"].squeeze(0)
    update(tgt_dict, target_name + ".weight", conv_weight)
    update(tgt_dict, target_name + ".affine.weight",     src_dict[source_name + ".conv.modulation.weight"])
    update(tgt_dict, target_name + ".affine.bias",       src_dict[source_name + ".conv.modulation.bias"])
    update(tgt_dict, target_name + ".noise_strength", src_dict[source_name + ".noise.weight"].squeeze())
    update(tgt_dict, target_name + ".bias",           src_dict[source_name + ".activate.bias"].squeeze())

def convert_torgb(tgt_dict, src_dict, target_name, source_name):
    update(tgt_dict, target_name + ".weight",        src_dict[source_name + ".conv.weight"].squeeze(0).squeeze(0))
    update(tgt_dict, target_name + ".affine.weight", src_dict[source_name + ".conv.modulation.weight"])
    update(tgt_dict, target_name + ".affine.bias",   src_dict[source_name + ".conv.modulation.bias"])
    update(tgt_dict, target_name + ".bias",          src_dict[source_name + ".bias"].squeeze())

def convert_dense(tgt_dict, src_dict, target_name, source_name):
    update(tgt_dict, target_name + ".weight", src_dict[source_name + ".weight"])
    update(tgt_dict, target_name + ".bias",   src_dict[source_name + ".bias"])

def update_D(src_dict, tgt_dict, size, n_mlp):
    log_size = int(math.log(size, 2))
    convert_conv(tgt_dict, src_dict, f"{size}x{size}/FromRGB", "convs.0")
    conv_i = 1
    pbar = ProgressBar(log_size-1)
    for i in range(log_size-2, 0, -1):
        reso = 4 * 2 ** i
        convert_conv(tgt_dict, src_dict, f"{reso}x{reso}/Conv0",      f"convs.{conv_i}.conv1")
        convert_conv(tgt_dict, src_dict, f"{reso}x{reso}/Conv1_down", f"convs.{conv_i}.conv2", start=1)
        convert_conv(tgt_dict, src_dict, f"{reso}x{reso}/Skip",       f"convs.{conv_i}.skip",  start=1, bias=False)
        conv_i += 1
        pbar.upd()
    convert_conv(tgt_dict, src_dict,  f"4x4/Conv",   "final_conv")
    convert_dense(tgt_dict, src_dict, f"4x4/Dense0", "final_linear.0")
    convert_dense(tgt_dict, src_dict, f"Output",     "final_linear.1")
    pbar.upd()

def update_G(src_dict, tgt_dict, size, n_mlp):
    log_size = int(math.log(size, 2))

    pbar = ProgressBar(n_mlp + log_size-2 + log_size-2 + (log_size-2)*2+1 + 2)
    for i in range(n_mlp):
        convert_dense(tgt_dict, src_dict, f"mapping.fc{i}", f"style.{i+1}")
        pbar.upd()
    update(tgt_dict, "synthesis.b4.const", src_dict["input.input"].squeeze(0))
    convert_torgb(tgt_dict, src_dict, "synthesis.b4.torgb", "to_rgb1")
    pbar.upd()

    for i in range(log_size-2):
        reso = 4 * 2 ** (i+1)
        convert_torgb(tgt_dict, src_dict, f"synthesis.b{reso}.torgb", f"to_rgbs.{i}")
        pbar.upd()
    convert_modconv(tgt_dict, src_dict, "synthesis.b4.conv1", "conv1")
    pbar.upd()

    conv_i = 0
    for i in range(log_size-2):
        reso = 4 * 2 ** (i+1)
        convert_modconv(tgt_dict, src_dict, f"synthesis.b{reso}.conv0", f"convs.{conv_i}")
        convert_modconv(tgt_dict, src_dict, f"synthesis.b{reso}.conv1", f"convs.{conv_i + 1}")
        conv_i += 2
        pbar.upd()

    for i in range(0, (log_size-2) * 2 + 1):
        reso = 4 * 2 ** (math.ceil(i/2))
        update(tgt_dict, f"synthesis.b{reso}.conv{(i+1)%2}.noise_const", src_dict[f"noises.noise_{i}"].squeeze())
        pbar.upd()

    src_kernels = [k for k in src_dict.keys() if 'kernel' in k]
    src_kernel = src_dict[src_kernels[0]]
    tgt_kernels = [k for k in tgt_dict.keys() if 'resample_filter' in k] # [0]
    for tgt_k in tgt_kernels:
        update(tgt_dict, tgt_k, src_kernel/4)

def load_net_from_pkl(path):
    tgt_net = load_pkl(path)
    Gs = tgt_net['G_ema']
    tgt_dict = tgt_net['G_ema'].state_dict()
    n_mlp = len([l for l in tgt_dict.keys() if l.startswith('mapping.fc')]) // 2
    size = tgt_net['G_ema'].img_resolution
    return tgt_net, size, n_mlp

if __name__ == "__main__":
    args = get_args()

    tgt_net, size, n_mlp = load_net_from_pkl(args.model_pkl)
    tgt_dict = tgt_net['G_ema'].state_dict()
    src_dict = torch.load(args.model_pt)
    update_G(src_dict['g_ema'], tgt_dict, size, n_mlp)
    
    out_name = args.model_pt.replace('.pt', '.pkl')
    save_pkl(dict(G_ema=tgt_net['G_ema']), out_name)


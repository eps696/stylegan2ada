# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
"""Train a GAN using the techniques described in the paper
"Training Generative Adversarial Networks with Limited Data"."""

import os
import click
import re
import json
import tempfile
from imageio import imread

import torch
from torch_utils import training_stats
from torch_utils import custom_ops

import dnnlib
from training import training_loop

from util.utilgan import calc_init_res, basename, file_list

class UserError(Exception):
    pass

def setup_training_loop_kwargs(
    # data
    data       = None, # Training dataset (required): <path>
    resume     = None, # Load previous network: 'noresume' (default), 'ffhq256', 'ffhq512', 'ffhq1024', 'celebahq256', 'lsundog256', <file>, <url>
    mirror     = None, # Augment dataset with x-flips: <bool>, default = False
    cond       = None, # Train conditional model based on dataset labels: <bool>, default = False
    # training
    cfg        = None, # Base config: 'auto' (default), 'stylegan2', 'paper256', 'paper512', 'paper1024', 'cifar'
    batch      = None, # Override batch size: <int>
    lod_kimg   = None, # Training duration per LOD/layer: <int>
    kimg       = None, # Override training duration: <int>
    snap       = None, # Snapshot interval: <int>, default = 5 ticks
    gamma      = None, # Override R1 gamma: <float>
    freezed    = None, # Freeze-D: <int>, default = 0 discriminator layers
    seed       = None, # Random seed: <int>, default = 0
    # d augment
    aug        = None, # Augmentation mode: 'ada' (default), 'noaug', 'fixed'
    p          = None, # Specify p for 'fixed' (required): <float>
    target     = None, # Override ADA target for 'ada': <float>, default = depends on aug
    augpipe    = None, # Augmentation pipeline: 'blit', 'geom', 'color', 'filter', 'noise', 'cutout', 'bg', 'bgc' (default), ..., 'bgcfnc'

    # general & perf options 
    gpus       = None, # Number of GPUs: <int>, default = 1 gpu
    fp32       = None, # Disable mixed-precision training: <bool>, default = False
    nhwc       = None, # Use NHWC memory format with FP16: <bool>, default = False
    workers    = None, # Override number of DataLoader workers: <int>, default = 3
    nobench    = None, # Disable cuDNN benchmarking: <bool>, default = False
):
    args = dnnlib.EasyDict()
    args.G_kwargs = dnnlib.EasyDict(class_name='training.networks.Generator', z_dim=512, w_dim=512, mapping_kwargs=dnnlib.EasyDict(), synthesis_kwargs=dnnlib.EasyDict())
    args.D_kwargs = dnnlib.EasyDict(class_name='training.networks.Discriminator', block_kwargs=dnnlib.EasyDict(), mapping_kwargs=dnnlib.EasyDict(), epilogue_kwargs=dnnlib.EasyDict())

    # General options: gpus, snap, seed
    # ------------------------------------------

    assert (gpus >= 1 and gpus & (gpus - 1) == 0), '--gpus must be a power of two'
    args.num_gpus = gpus

    assert snap > 1, '--snap must be at least 1'
    args.image_snapshot_ticks = snap
    args.network_snapshot_ticks = snap

    args.random_seed = seed

    # Dataset: data, cond, subset, mirror
    # -----------------------------------

    assert data is not None
    args.training_set_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=data, use_labels=True, max_size=None, xflip=False)
    args.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, num_workers=3, prefetch_factor=2)
    try:
        training_set = dnnlib.util.construct_class_by_name(**args.training_set_kwargs) # subclass of training.dataset.Dataset
        args.training_set_kwargs.resolution = res = training_set.resolution # be explicit about resolution
        args.training_set_kwargs.use_labels = training_set.has_labels # be explicit about labels
        args.training_set_kwargs.max_size = len(training_set) # be explicit about dataset size
# !!! custom init res
        image_shape = training_set.image_shape
        init_res = training_set.init_res
        res_log2 = training_set.res_log2
        desc = dataname = training_set.name
        del training_set # conserve memory
    except IOError as err:
        raise UserError(f'--data: {err}')

# !!! custom init res
    if list(init_res) == [4,4]: 
        desc += '-%d' % res
    else:
        print(' custom init resolution', init_res)
        args.G_kwargs.init_res = args.D_kwargs.init_res = list(init_res)
        desc += '-%dx%d' % (image_shape[2], image_shape[1])

# !!! 
    args.savenames = [desc.replace(dataname, 'snapshot'), desc]

    if cond:
        if not args.training_set_kwargs.use_labels:
            # raise UserError('--cond=True requires labels specified in dataset.json')
            raise UserError(' put images in flat subdirectories for conditional training')
        desc += '-cond'
    else:
        args.training_set_kwargs.use_labels = False

    if mirror:
        # desc += '-mirror'
        args.training_set_kwargs.xflip = True

    # Base config: cfg, gamma, kimg, batch
    # ------------------------------------

    desc += f'-{cfg}'
    if gpus > 1: desc += f'{gpus:d}'

    cfg_specs = {
        'auto': dict(ramp=0.05, map=2), # mapping_layers = 2 for uptrain (why?)
        'eps':  dict(lrate=0.001, ema=10, ramp=0.05, map=2), # populated based on 'gpus' and 'res'
        'big':  dict(mb=4, fmaps=1, lrate=0.002, gamma=10, ema=10, ramp=None, map=8), # aydao etc
    }

    assert cfg in cfg_specs
    spec = dnnlib.EasyDict(cfg_specs[cfg])
    if cfg == 'auto':
        # spec.mb = max(min(gpus * min(4096 // res, 32), 64), gpus) # keep gpu memory consumption at bay
        spec.mb = max(min(gpus * min(3072 // res, 32), 64), gpus) # keep gpu memory consumption at bay
        spec.fmaps = 1 if res >= 512 else 0.5
        spec.lrate = 0.002 if res >= 1024 else 0.0025
        spec.gamma = 0.0002 * (res ** 2) / spec.mb # heuristic formula
        spec.ema = spec.mb * 10 / 32
    elif cfg == 'eps':
        spec.mb = max(min(gpus * min(3072 // res, 32), 64), gpus)
        spec.fmaps = 1 if res >= 512 else 0.5
        spec.gamma = 0.00001 * (res ** 2) / spec.mb # !!! my mb 3~4 instead of 32~64
    spec.ref_gpus = gpus
    spec.mbstd = spec.mb // gpus # min(spec.mb // gpus, 4) # other hyperparams behave more predictably if mbstd group size remains fixed

    args.G_kwargs.synthesis_kwargs.channel_base = args.D_kwargs.channel_base = int(spec.fmaps * 32768) # TWICE MORE than sg2 on tf !!!
    args.G_kwargs.synthesis_kwargs.channel_max = args.D_kwargs.channel_max = 512
    args.G_kwargs.mapping_kwargs.num_layers = spec.map
    args.G_kwargs.synthesis_kwargs.num_fp16_res = args.D_kwargs.num_fp16_res = 4 # enable mixed-precision training
    args.G_kwargs.synthesis_kwargs.conv_clamp = args.D_kwargs.conv_clamp = 256 # clamp activations to avoid float16 overflow
    args.D_kwargs.epilogue_kwargs.mbstd_group_size = spec.mbstd

    args.G_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', lr=spec.lrate, betas=[0,0.99], eps=1e-8)
    args.D_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', lr=spec.lrate, betas=[0,0.99], eps=1e-8)
    args.loss_kwargs = dnnlib.EasyDict(class_name='training.loss.StyleGAN2Loss', r1_gamma=spec.gamma)

    # use per-layer duration lod_kimg, or override with total kimg
    args.total_kimg = kimg if (kimg is not None and kimg >= 1) else lod_kimg * res_log2 * 3 # ~ ProGAN *1.5
    
    args.batch_size = spec.mb
    args.batch_gpu = spec.mb // spec.ref_gpus
    args.ema_kimg = spec.ema
    args.ema_rampup = spec.ramp

    """ 
    if 'div' in cfg: # for diverse datasets a la cifar
        args.loss_args.pl_weight = 0 # disable path length regularization
        args.G_args.style_mixing_prob = None # disable style mixing
        args.D_args.architecture = 'orig' # disable residual skip connections

    if cfg == 'big': # > 100k img
        # disable path length and style mixing regularization
        args.loss_args.pl_weight = 0
        args.G_args.style_mixing_prob = None
        # double generator capacity
        args.G_kwargs.synthesis_kwargs.channel_base = args.D_kwargs.channel_base = 32 << 10 # if double for torch, should be 64 << 10 ??
        args.G_kwargs.synthesis_kwargs.channel_max  = args.D_kwargs.channel_max  = 1024
        # enable top k training
        args.loss_args.G_top_k = True # drop bad samples
        # args.loss_args.G_top_k_gamma = 0.99 # takes ~70% of full training from scratch to decay to 0.5
        # args.loss_args.G_top_k_gamma = 0.9862 # takes 12500 kimg to decay to 0.5 (~1/2 of total_kimg when training from scratch)
        args.loss_args.G_top_k_gamma = 0.9726 # takes 6250 kimg to decay to 0.5 (~1/4 of total_kimg when training from scratch)
        args.loss_args.G_top_k_frac = 0.5
        # reduce in-memory size, you need a BIG GPU for big model
        args.batch_gpu = 4 # probably will need to set this pretty low with such a large G, higher values work better for top-k training though
        args.G_kwargs.synthesis_kwargs.num_fp16_res = 6 # making more layers fp16 can help as well
    """

    if gamma is not None:
        assert gamma >= 0, '--gamma must be non-negative'
        desc += f'-gamma{gamma:g}'
        args.loss_kwargs.r1_gamma = gamma

    if batch is not None:
        assert (batch >= 1 and batch % gpus == 0), '--batch must be at least 1 and divisible by --gpus'
        desc += f'-batch{batch}'
        args.batch_size = batch
        args.batch_gpu = batch // gpus

    # Discriminator augmentation: aug, p, target, augpipe
    # ---------------------------------------------------

    if aug != 'ada': desc += f'-{aug}'

    if aug == 'ada':
        args.ada_target = 0.6
    elif aug == 'noaug':
        pass
    elif aug == 'fixed':
        if p is None:
            raise UserError(f'--aug={aug} requires specifying --p')
    else:
        raise UserError(f'--aug={aug} not supported')

    if p is not None:
        assert aug == 'fixed', '--p can only be specified with --aug=fixed'
        assert 0 <= p <= 1, '--p must be between 0 and 1'
        desc += f'-p{p:g}'
        args.augment_p = p

    if target is not None:
        assert aug == 'ada', '--target can only be specified with --aug=ada'
        assert 0 <= target <= 1, '--target must be between 0 and 1'
        desc += f'-target{target:g}'
        args.ada_target = target

    if augpipe is None and aug == 'noaug':
        raise UserError('--augpipe cannot be specified with --aug=noaug')
    desc += f'-{augpipe}'
    args.augpipe = augpipe

    augpipe_specs = {
        'blit':   dict(xflip=1, rotate90=1, xint=1),
        'geom':   dict(scale=1, rotate=1, aniso=1, xfrac=1),
        'color':  dict(brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1),
        'filter': dict(imgfilter=1),
        'noise':  dict(noise=1),
        'cutout': dict(cutout=1),
        'bg':     dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1),
        'bgc':    dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1),
        'bgcf':   dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1, imgfilter=1),
        'bgcfn':  dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1, imgfilter=1, noise=1),
        'bgcfnc': dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1, imgfilter=1, noise=1, cutout=1),
# !!!
        'bgf_cnc':  dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, contrast=0.23, imgfilter=1, noise=0.11, cutout=0.11),
        'gf_bnc':   dict(xflip=.5, xint=.5, scale=1, rotate=1, aniso=1, xfrac=1, rotate_max=.25, imgfilter=1, noise=.5, cutout=.5), # aug0
        # 'bg_cfnc':  dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, contrast=0.11, imgfilter=0.11, noise=0.11, cutout=0.11),
        # 'aug1':     dict(xflip=.25, xint=.25, scale=.5, rotate=.5, aniso=.5, xfrac=.5, rotate_max=.25, imgfilter=.25, cutout=.25),
        # 'bg2':      dict(xflip=.25, xint=.25, scale=.5, rotate=.5, aniso=.5, xfrac=.5, rotate_max=.25),
        # 'augg':     dict(scale=.5, rotate=.5, aniso=.5, xfrac=.5, rotate_max=.25, imgfilter=.25),
        # 'aug2':     dict(xflip=.25, rotate90=.25, xint=.25, scale=.5, rotate=.5, aniso=.5, xfrac=.5, rotate_max=.25, imgfilter=.25, noise=.25, cutout=.25),
        # 'bgg2':     dict(xflip=.5, rotate90=.25, xint=.5, scale=1, rotate=1, aniso=1, xfrac=.5, rotate_max=.25, imgfilter=.25, noise=.25, cutout=.25),
        # 'bgg3':     dict(xflip=.5, rotate90=.25, scale=1, rotate=1, aniso=1, rotate_max=.25, imgfilter=.25, noise=.25, cutout=.25),
        # 'gg':       dict(scale=.25, rotate=.25, aniso=.25, xfrac=.25, rotate_max=.25, imgfilter=.25, cutout=.25),
    }

    assert augpipe in augpipe_specs, ' unknown augpipe specs: %s' % augpipe
    if aug != 'noaug':
        args.augment_kwargs = dnnlib.EasyDict(class_name='training.augment.AugmentPipe', **augpipe_specs[augpipe])

    # Transfer learning: resume, freezed
    # ----------------------------------

    resume_specs = {
        'ffhq256':     'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/ffhq-res256-mirror-paper256-noaug.pkl',
        'ffhq512':     'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/ffhq-res512-mirror-stylegan2-noaug.pkl',
        'ffhq1024':    'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/ffhq-res1024-mirror-stylegan2-noaug.pkl',
        'celebahq256': 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/celebahq-res256-mirror-paper256-kimg100000-ada-target0.5.pkl',
        'lsundog256':  'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/lsundog-res256-paper256-kimg100000-noaug.pkl',
    }

    if resume is None:
        pass
    elif resume in resume_specs:
        # desc += f'-resume{resume}'
        args.resume_pkl = resume_specs[resume] # predefined url
    else:
        # desc += '-resumecustom'
        args.resume_pkl = resume # custom path or url

    if resume is not None:
        args.ada_kimg = 100 # make ADA react faster at the beginning
        args.ema_rampup = None # disable EMA rampup

    if freezed is not None:
        assert freezed >= 0, '--freezed must be non-negative'
        desc += f'-freezed{freezed:d}'
        args.D_kwargs.block_kwargs.freeze_layers = freezed

    # Performance options: fp32, nhwc, nobench, workers
    # -------------------------------------------------

    if fp32:
        args.G_kwargs.synthesis_kwargs.num_fp16_res = args.D_kwargs.num_fp16_res = 0
        args.G_kwargs.synthesis_kwargs.conv_clamp = args.D_kwargs.conv_clamp = None

    if nhwc:
        args.G_kwargs.synthesis_kwargs.fp16_channels_last = args.D_kwargs.block_kwargs.fp16_channels_last = True

    if nobench:
        args.cudnn_benchmark = False

    if workers is not None:
        assert workers >= 1, '--workers must be at least 1'
        args.data_loader_kwargs.num_workers = workers

    return desc, args

def subprocess_fn(rank, args, temp_dir):
    dnnlib.util.Logger(file_name=os.path.join(args.run_dir, 'log.txt'), file_mode='a', should_flush=True)

    # Init torch.distributed.
    if args.num_gpus > 1:
        init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
        if os.name == 'nt':
            init_method = 'file:///' + init_file.replace('\\', '/')
            torch.distributed.init_process_group(backend='gloo', init_method=init_method, rank=rank, world_size=args.num_gpus)
        else:
            init_method = f'file://{init_file}'
            torch.distributed.init_process_group(backend='nccl', init_method=init_method, rank=rank, world_size=args.num_gpus)

    # Init torch_utils.
    sync_device = torch.device('cuda', rank) if args.num_gpus > 1 else None
    training_stats.init_multiprocessing(rank=rank, sync_device=sync_device)
    if rank != 0:
        custom_ops.verbosity = 'none'

    # Execute training loop.
    training_loop.training_loop(rank=rank, **args)

class CommaSeparatedList(click.ParamType):
    name = 'list'
    def convert(self, value, param, ctx):
        _ = param, ctx
        if value is None or value.lower() == 'none' or value == '':
            return []
        return value.split(',')

@click.command()
@click.pass_context
@click.option('--train_dir', default='train', help='Root directory for training results [default: train]', metavar='DIR')
# data
@click.option('--data', help='Training data (directory or zip)', metavar='PATH', required=True)
@click.option('--resume', help='Resume training [default: None]', metavar='PKL')
@click.option('--mirror', default=True, type=bool, help='Enable dataset x-flips [default: true]', metavar='BOOL')
@click.option('--cond', is_flag=True, help='Train conditional model based on dataset labels [default: false]', metavar='BOOL')
# training
@click.option('--cfg', default='auto', help='Base config [default: auto]'))
@click.option('--batch', type=int, help='Override batch size', metavar='INT')
@click.option('--lod_kimg', default=30, type=int, help='Per layer training duration', metavar='INT')
@click.option('--kimg', type=int, help='Override total training duration', metavar='INT')
@click.option('--snap', default=5, type=int, help='Snapshot interval [default: 5 ticks]', metavar='INT')
@click.option('--gamma', type=float, help='Override R1 gamma')
@click.option('--freezed', type=int, help='Freeze-D [default: 0 layers]', metavar='INT')
@click.option('--seed', default=0, type=int, help='Random seed [default: 0]', metavar='INT')
# Discriminator augmentation.
@click.option('--aug', default='ada', help='Augmentation mode [default: ada]', type=click.Choice(['noaug', 'ada', 'fixed']))
@click.option('--p', type=float, help='Augmentation probability for --aug=fixed')
@click.option('--target', type=float, help='ADA target value for --aug=ada')
# @click.option('--augpipe', default='bgc', help='Augmentation pipeline', type=click.Choice(['blit', 'geom', 'color', 'filter', 'noise', 'cutout', 'bg', 'bgc', 'bgcf', 'bgcfn', 'bgcfnc']))
@click.option('--augpipe', default='gf_bnc', help='Augmentation pipeline')
# misc perf
@click.option('--gpus', default=1, help='Number of GPUs to use [default: 1]', type=int, metavar='INT')
@click.option('--fp32', is_flag=True, help='Disable mixed-precision training', metavar='BOOL')
@click.option('--nhwc', is_flag=True, help='Use NHWC memory format with FP16', metavar='BOOL')
@click.option('--workers', type=int, help='Override number of DataLoader workers', metavar='INT')
@click.option('--nobench', is_flag=True, help='Disable cuDNN benchmarking', metavar='BOOL')
@click.option('-n', '--dry-run', is_flag=True, help='Print training options and exit')


def main(ctx, train_dir, dry_run, **config_kwargs):
    dnnlib.util.Logger(should_flush=True)

    # Setup training options.
    try:
        run_desc, args = setup_training_loop_kwargs(**config_kwargs)
    except UserError as err:
        ctx.fail(err)

    # Pick output directory.
    prev_run_dirs = []
    if os.path.isdir(train_dir):
        prev_run_dirs = [x for x in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, x))]
    prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id = max(prev_run_ids, default=-1) + 1
    args.run_dir = os.path.join(train_dir, f'{cur_run_id:03d}-{run_desc}')
    assert not os.path.exists(args.run_dir)

    # Print options.
    # print(' Training options:')
    # print(json.dumps(args, indent=2))
    print(' Train dir: ', args.run_dir)
    print(' Dataset:   ', args.training_set_kwargs.path)
    print(' Resolution:', args.training_set_kwargs.resolution)
    print(' Batch:     ', args.batch_gpu)
    print(' Length:     %d kimg' % args.total_kimg)
    print(' Augment:   ', args.augpipe)
    print(' Num images:', args.training_set_kwargs.max_size, '(+mirror)' if args.training_set_kwargs.xflip else '')

    if args.training_set_kwargs.use_labels:
        print(' Conditional model')
    if args.num_gpus > 1:
        print(' GPUs:       %d' % args.num_gpus)
    if dry_run:
        print('Dry run; exiting.')
        return

    # Create output directory.
    os.makedirs(args.run_dir)
    with open(os.path.join(args.run_dir, 'training_options.json'), 'wt') as f:
        json.dump(args, f, indent=2)

    # Launch processes.
    # print('Launching processes...')
    torch.multiprocessing.set_start_method('spawn')
    with tempfile.TemporaryDirectory() as temp_dir:
        if args.num_gpus == 1:
            subprocess_fn(rank=0, args=args, temp_dir=temp_dir)
        else:
            torch.multiprocessing.spawn(fn=subprocess_fn, args=(args, temp_dir), nprocs=args.num_gpus)


if __name__ == "__main__":

    # workaround for multithreading in jupyter console
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"

    main() # pylint: disable=no-value-for-parameter


import sys
import os
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))               # tasks/sdf/
sys.path.insert(0, os.path.join(_THIS_DIR, 'lib'))                   # bacon helpers
sys.path.insert(0, os.path.dirname(os.path.dirname(_THIS_DIR)))      # FLAIR root → modules.bla

from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
from torch.utils.data import DataLoader
import configargparse
import dataio
import utils
import training
import loss_functions
from functools import partial

torch.set_num_threads(8)

p = configargparse.ArgumentParser()

# config file, output directories
p.add('-c', '--config', required=False, is_config_file=True,
      help='Path to config file.')
p.add_argument('--logging_root', type=str, default='../logs',
               help='root for logging')
p.add_argument('--experiment_name', type=str, required=True,
               help='subdirectory in logging_root for checkpoints, summaries')



# general training
p.add_argument('--model_type', type=str, default='bla',
               help='options: mfn, siren, ff, finer, wire, nerfpe')
p.add_argument('--hidden_size', type=int, default=128,
               help='size of hidden layer')
p.add_argument('--hidden_layers', type=int, default=8)
p.add_argument('--lr', type=float, default=1e-4, help='learning rate')
p.add_argument('--num_steps', type=int, default=20000,
               help='number of training steps')
p.add_argument('--ckpt_step', type=int, default=0,
               help='step at which to resume training')
p.add_argument('--gpu', type=int, default=1, help='GPU ID to use')
p.add_argument('--seed', default=None,
               help='random seed for experiment reproducibility')

# mfn options
p.add_argument('--multiscale', action='store_true', default=False,
               help='use multiscale')
p.add_argument('--max_freq', type=int, default=512,
               help='The network-equivalent sample rate used to represent the signal.'
               + 'Should be at least twice the Nyquist frequency.')
p.add_argument('--input_scales', nargs='*', type=float, default=None,
               help='fraction of resolution growth at each layer')
p.add_argument('--output_layers', nargs='*', type=int, default=None,
               help='layer indices to output, beginning at 1')

# mlp options
p.add_argument('--w0', default=30, type=int,
               help='omega_0 parameter for siren')
p.add_argument('--pe_scale', default=5, type=float,
               help='positional encoding scale')
#
p.add_argument('--fbs', type=float, default=None, help='')

#rcg options

p.add_argument('--init_t', type=float, default=1.0,
               help='initial T parameter for RC-Gauss')
p.add_argument('--init_beta', type=float, default=0.05,
               help='initial beta parameter for RC-Gauss (non-trainable)')
p.add_argument('--init_zeta', type=float, default=1.0,
               help='initial zeta parameter for RC-Gauss')
p.add_argument('--sigma', type=float, default=30.0,
               help='initial Gaussian sigma parameter for RC-Gauss')
p.add_argument('--trainable', type=bool, default=True,
               help='whether RC-Gauss parameters are learnable')



# sdf model and sampling
p.add_argument('--num_pts_on', type=int, default=10000,
               help='number of on-surface points to sample')
p.add_argument('--coarse_scale', type=float, default=1e-1,
               help='laplacian scale factor for coarse samples')
p.add_argument('--fine_scale', type=float, default=1e-3,
               help='laplacian scale factor for fine samples')
p.add_argument('--coarse_weight', type=float, default=1e-2,
               help='weight to apply to coarse loss samples')

# data i/o
p.add_argument('--shape', type=str, default='bunny',
               help='name of point cloud shape in xyz format')
p.add_argument('--point_cloud_path', type=str,
               default='../data/armadillo.xyz',
               help='path for input point cloud')
p.add_argument('--num_workers', default=0, type=int,
               help='number of workers')

# tensorboard summary
p.add_argument('--steps_til_ckpt', type=int, default=50000,
               help='epoch frequency to save a checkpoint')
p.add_argument('--steps_til_summary', type=int, default=1000,
               help='epoch frequency to update tensorboard summary')
p.add_argument('--fast', action='store_true',
               help='use bla_float (real-valued, compile-compatible) + torch.compile')

opt = p.parse_args()


def main():

    print('--- Run Configuration ---')
    for k, v in vars(opt).items():
        print(k, v)

    train()


def train():

    opt.root_path = os.path.join(opt.logging_root, opt.experiment_name)
    utils.cond_mkdir(opt.root_path)

    if opt.seed:
        torch.manual_seed(int(opt.seed))
        np.random.seed(int(opt.seed))

    dataloader = init_dataloader(opt)

    model = init_model(opt)

    loss_fn, summary_fn = init_loss(opt)

    save_params(opt, model)

    training.train(model=model, train_dataloader=dataloader, steps=opt.num_steps,
                   lr=opt.lr, steps_til_summary=opt.steps_til_summary,
                   ckpt_step=opt.ckpt_step,
                   steps_til_checkpoint=opt.steps_til_ckpt,
                   model_dir=opt.root_path, loss_fn=loss_fn, summary_fn=summary_fn,
                   double_precision=False, clip_grad=True,
                   use_lr_scheduler=True, is_wire=(opt.model_type=='wire' or opt.model_type=='wf'))


def init_dataloader(opt):
    ''' load sdf dataloader via eikonal equation or fitting sdf directly '''

    sdf_dataset = dataio.MeshSDF(opt.point_cloud_path,
                                 num_samples=opt.num_pts_on,
                                 coarse_scale=opt.coarse_scale,
                                 fine_scale=opt.fine_scale)

    dataloader = DataLoader(sdf_dataset, shuffle=True,
                            batch_size=1, pin_memory=True,
                            num_workers=opt.num_workers)

    return dataloader


class BLAforSDF(torch.nn.Module):
    """Adapter that bridges FLAIR's BLA INR (tensor in/out) to bacon's
    dict-based forward interface used by training/loss code.
    use_fast=True swaps cfloat `bla` → real-valued `bla_float` (compile-compatible)."""
    def __init__(self, hidden_features, hidden_layers, scale,
                 init_T, init_beta, init_zeta, use_fast=False):
        super().__init__()
        if use_fast:
            from modules.bla_float import INR
        else:
            from modules.bla import INR
        self.inr = INR(
            in_features=3, hidden_features=hidden_features,
            hidden_layers=hidden_layers, out_features=1,
            scale=scale,
            init_T=init_T, init_beta=init_beta, init_zeta=init_zeta,
            use_sigmoid=False,
        )

    def forward(self, model_input):
        coords = model_input['coords']            # (B, N, 3) tensor
        pred = self.inr(coords)                   # (B, N, 1) tensor
        return {'model_in': model_input, 'model_out': pred}


def init_model(opt):
    '''Build INR model. BLA-only fork — uses FLAIR's modules/bla.py.'''
    if opt.model_type != 'bla':
        raise NotImplementedError(
            f"Only --model_type bla is supported. Got {opt.model_type!r}."
        )

    model = BLAforSDF(
        hidden_features=opt.hidden_size,
        hidden_layers=opt.hidden_layers,
        scale=opt.sigma,
        init_T=opt.init_t,
        init_beta=opt.init_beta,
        init_zeta=opt.init_zeta,
        use_fast=getattr(opt, 'fast', False),
    )

    if getattr(opt, 'fast', False):
        from modules.speed import setup_fast_env, maybe_compile
        setup_fast_env()
        model = maybe_compile(model, enabled=True)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'Num. Parameters: {params}')
    model.cuda()

    # if resuming model training
    if opt.ckpt_step:
        opt.num_steps -= opt.ckpt_step  # steps remaning to train
        if opt.num_steps < 1:
            raise ValueError('ckpt_epoch must be less than num_epochs')
        print(opt.num_steps)

        pth_file = '{}/checkpoints/model_step_{}.pth'.format(opt.root_path,
                                                             str(opt.ckpt_step).zfill(4))
        model.load_state_dict(torch.load(pth_file))

    return model


def init_loss(opt):
    ''' define loss, summary functions given expmt configs '''

    if opt.multiscale:
        summary_fn = utils.write_multiscale_sdf_summary
        loss_fn = partial(loss_functions.multiscale_overfit_sdf,
                          coarse_loss_weight=opt.coarse_weight)
    else:
        summary_fn = utils.write_sdf_summary
        loss_fn = partial(loss_functions.overfit_sdf,
                          coarse_loss_weight=opt.coarse_weight)

    return loss_fn, summary_fn


def save_params(opt, model):

    # Save command-line parameters log directory.
    p.write_config_file(opt, [os.path.join(opt.root_path, 'config.ini')])
    with open(os.path.join(opt.root_path, "params.txt"), "w") as out_file:
        out_file.write('\n'.join(["%s: %s" % (key, value) for key, value in vars(opt).items()]))

    # Save text summary of model into log directory.
    with open(os.path.join(opt.root_path, "model.txt"), "w") as out_file:
        out_file.write(str(model))


if __name__ == '__main__':
    main()

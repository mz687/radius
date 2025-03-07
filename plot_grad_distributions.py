import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from datashader.mpl_ext import dsshow
from itertools import repeat
from multiprocessing import Pool
# from ray.util.multiprocessing.pool import Pool
from tqdm import tqdm
from collections import OrderedDict as odict
import torch
import datashader as ds
import math
import shutil
import seaborn as sns

from megatron import get_args
from megatron.model import GPTModel
from megatron.initialize import initialize_megatron
from megatron.model import ModelType
from megatron.training import setup_model_and_optimizer

import multiprocessing.pool as mpp
from tqdm.contrib.concurrent import process_map

# from mpi4py import MPI
# comm = MPI.COMM_WORLD

import torch

def istarmap(self, func, iterable, chunksize=1):
  """starmap-version of imap
  """
  self._check_running()
  if chunksize < 1:
      raise ValueError(
          "Chunksize must be 1+, not {0:n}".format(
              chunksize))

  task_batches = mpp.Pool._get_tasks(func, iterable, chunksize)
  result = mpp.IMapIterator(self)
  self._taskqueue.put(
      (
          self._guarded_task_generation(result._job,
                                        mpp.starmapstar,
                                        task_batches),
          result._set_length
      ))
  return (item for chunk in result for item in chunk)

# def get_args_parser():
#   parser = argparse.ArgumentParser('Plot topK gradient distribution', add_help=False)
#   parser.add_argument('--model', default='gpt', type=str,
#                       help='model name (e.g. gpt, bert). ONLY supports GPT models now!')
#   parser.add_argument('--dir', default='/work/09308/zhengmk/ViT/mae/results/topk', type=str,
#                       help='Input topK checkpoints')
#   parser.add_argument('--num-layers', dest='num_layers', type=int, required=True)
#   parser.add_argument('--hidden-size', dest='hidden_size', type=int, required=True)
#   parser.add_argument('--out_dir', default='/work/09308/zhengmk/optimus-cc/optimus-cc/jaeyong-song-Optimus-CC-6249c49/plot_gradient_distributions/plots', type=str)
#   parser.add_argument('--freq', default=100, type=int)
#   parser.add_argument('--num_process', default=10, type=int)
#   parser.add_argument('--bucket_size', default=1, type=int)
#   parser.set_defaults(norm_pix_loss=False)
#   return parser


def model_provider(pre_process=True, post_process=True):
    """Build the model."""

    print('building GPT model ...')
    model = GPTModel(
        num_tokentypes=0,
        parallel_output=True,
        pre_process=pre_process,
        post_process=post_process
    )
    return model


def get_model_num_param(model_provider_func, model_type=ModelType.encoder_or_decoder)->dict:
  args = get_args()
  args.model_type = model_type
  
  model, optimizer, lr_scheduler = setup_model_and_optimizer(model_provider,
                                                               model_type)
  
  total_num_param = 0
  layer_params = {}
  # model = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)
  for module in model:
    for index, (name, parameter) in enumerate(module.named_parameters()):
      name = name.replace("module.module","module")
      if parameter.requires_grad:
        layer_params[name] = parameter.numel()
        total_num_param += layer_params[name]

  model_parameters = pd.DataFrame({'layer name':layer_params.keys(),'numel':layer_params.values()})
  model_parameters.to_csv(os.path.join(args.out_dir,"model_parameters.csv"),sep=',',index=False)
  return layer_params

def plot_layers(args):
  out_folder_name = os.path.join(args.out_dir, f"num_buckets_{args.num_buckets}")
  # if os.path.exists(out_folder_name):
  #   shutil.rmtree(out_folder_name)
  os.makedirs(out_folder_name, exist_ok=True)
  layer_params = get_model_num_param(model_provider)

  os.chdir(args.dir)
  # only show the results of first 100 (x100) steps
  steps = [step for step in os.listdir() if int(step.replace("iter_","")) % 100 == 0]
  print("Total num of steps: ", len(steps))
  folders = sorted(steps, key=lambda x: int(x.replace("iter_","")))
  world_size = int(os.environ["WORLD_SIZE"])
  world_rank = torch.distributed.get_rank()

  # Distribute the job of plotting gradient graph for 197 layers onto multiple nodes
  tot_num_plots = len(layer_params.keys())
  num_layers_per_node = math.ceil(tot_num_plots/world_size)
  layers_for_cur_node = list(layer_params.keys())[world_rank*num_layers_per_node: (world_rank+1)*num_layers_per_node]

  inputs = zip(repeat(args), repeat(folders), layers_for_cur_node, repeat(layer_params))
  with Pool(args.num_process) as pool:
    res = list(tqdm(pool.istarmap(plot_each_layer, inputs), total=len(layers_for_cur_node), desc="Plotting") )
  torch.distributed.barrier()

def plot_each_layer(args, folders:list, layer_name:str, size_dict:dict):
  args=get_args()
  
  distributions = {}

  # bucket_size = args.bucket_size if size_dict[layer_name] > 1e5 else 1

  bucket_size = None
  num_buckets = args.num_buckets
  if size_dict[layer_name] < num_buckets:
    bucket_size = 1
  else:
    bucket_size = size_dict[layer_name] // num_buckets

  for folder in folders:
    os.chdir(os.path.join(args.dir, folder))
    ckpt_files = sorted(os.listdir())
    cur_step = int(folder.replace("iter_",""))
    for ckpt in ckpt_files:
      if layer_name in ckpt:
        indices = torch.load(os.path.join(os.getcwd(), ckpt), map_location='cpu')
        distributions[cur_step] = pd.DataFrame(data=odict([('x', cur_step//args.freq),
                                                          ('y', np.array([int(i//bucket_size) for i in indices.numpy()]))]))
        break
  
  assert(len(folders) == len(distributions))
  df = pd.concat(distributions, ignore_index=True)
  steps = len(folders)
  high = size_dict[layer_name]

  # plot(args, df, bucket_size, steps, high, layer_name)
  plot_sns(args, df, bucket_size, steps, int(high // bucket_size), layer_name)
  
  return layer_name

def plot_sns(args, df, bucket_size, steps, high, layer_name):
  cmap=["#F8F8FF", "#CAC9CD", "#9B9A9C", "#6D6A6A", "#3E3B39", "#100C07"]
  res = df.value_counts(['x','y']).to_frame().reset_index()
  res['value'] = res[0] # count result is stored in column 0
  # res['value'] = res['count']
  
  # res.drop(0, axis=1, inplace=True)
  os.makedirs(os.path.join(args.dir, "../csv_data"), exist_ok=True)
  res.to_csv(os.path.join(args.dir, "../csv_data", f"{layer_name}.csv"), index=False)

  fig = plt.figure(figsize=(12,8), dpi=200)
  plt.rcParams.update({'font.size': 15})
  plt.rcParams["font.family"] = "Times New Roman"

  with sns.axes_style("white"):
    # res.to_csv(os.path.join(os.path.join(args.out_dir, f"bucket_size_{args.bucket_size}"), f"{layer_name}.csv"), index=False)
    ax = sns.heatmap(res.pivot(index='y',columns='x', values='value'), 
               cmap=cmap, cbar=False, yticklabels=high//15, xticklabels=10)
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.xlabel(rf"Steps (x$10^{int(np.log10(args.freq))}$)",fontsize=18)
    # plt.yticks([i for i in range(0, high, high//5)])
    if bucket_size > 100:
      plt.ylabel(rf"Indices (x$10^{int(np.log10(bucket_size))}$)",fontsize=18)
      plt.title(rf"{layer_name.replace('_indices', '')} (Bucket Size = $10^{int(np.log10(bucket_size))}$)", fontsize=20)
    else:
      plt.ylabel(f"Indices (x{bucket_size})",fontsize=18)
      plt.title(rf"{layer_name.replace('_indices', '')} (Bucket Size = {bucket_size})", fontsize=20)
    
    plt.savefig(os.path.join(os.path.join(args.out_dir, f"num_buckets_{args.num_buckets}"), f"{layer_name}.png"), bbox_inches='tight')
    plt.close(fig)


def plot(args, df, bucket_size, steps, high, layer_name):

  fig = plt.figure(figsize=(10, 10), dpi=200)
  font_family = {'fontname':'Times New Roman'}
  ax = plt.subplot(111)
  # fig, ax = plt.subplots()

  cmap = plt.get_cmap('binary')
  # cmap = 'inferno'
  # cmap =['lightgray', 'black']
  # cmap = ["#F8F8FF", "#CAC9CD", "#9B9A9C", "#6D6A6A", "#3E3B39", "#100C07"]
  artist = dsshow(df, 
                  ds.Point('x', 'y'),
                  ds.count(),
                  vmin=0,
                  cmap=cmap, 
                  ax=ax,
                  plot_width=steps,
                  plot_height=high//bucket_size,
                  y_range=(-1, high//bucket_size),
                  x_range=(0, steps),
                  norm='linear',
                  aspect='auto')

  plt.colorbar(artist, ax=ax, orientation='vertical', label='Count')
  plt.ylabel(f"Indices (x$10^{int(np.log10(bucket_size))}$)",fontsize=18)
  plt.xlabel(rf"Steps (x$10^{int(np.log10(args.freq))}$)",fontsize=18)
  plt.title(rf"Top 1% Gradient Distribution for {layer_name} (bucket size = $10^{int(np.log10(bucket_size))}$)", fontsize=18)
  plt.xticks([i for i in range(0, steps, 100)])
  plt.yticks([i for i in range(0, high//bucket_size, (high//bucket_size)//5)])
  # plt.grid(True)
  plt.savefig(os.path.join(os.path.join(args.out_dir, f"num_buckets_{args.num_buckets}"), f"{layer_name}.png"), bbox_inches='tight')
  plt.close(fig)

if __name__ == '__main__':
  mpp.Pool.istarmap = istarmap
  
  initialize_megatron(extra_args_provider=None,
                        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
                        allow_no_cuda=True)
  
  args = get_args()
  args.dir = "/pscratch/sd/m/mzheng/optimus-cc/grad_distr/GPT2_345M_range_buckets_EF_discont_buffers_correction_clip_lr_0.00015_density_1_range_10000_update_interval_100_warmup_method_Dense_warmup_threshold_10000/start_from_80000/GPT_345M_range_add_residual_to_org_only_gradients"
  args.out_dir="/pscratch/sd/m/mzheng/optimus-cc/grad_distr/GPT2_345M_range_buckets_EF_discont_buffers_correction_clip_lr_0.00015_density_1_range_10000_update_interval_100_warmup_method_Dense_warmup_threshold_10000/start_from_80000/GPT_345M_range_add_residual_to_org_only_gradients_plots"
  args.model="gpt"
  args.freq=100
  args.num_process=2
  args.num_buckets=5000
  
  assert args.model.lower() == 'gpt'

  plot_layers(args)
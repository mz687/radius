import subprocess
import os
import argparse
from datetime import datetime
import torch.distributed as dist
from tqdm import tqdm
import pandas as pd
import gc
import sys


def parse_args():
  parser = argparse.ArgumentParser(description="Get Nodes' hostnames")

  parser.add_argument('--output_file',
                        default='./NODEFILE',
                        type=str,
                        help='where the NODEFILE info will be saved')
  parser.add_argument("--num_gpus_per_node",
                      default=4,
                      type=int,
                      help='number of gpus per compute node')
  parser.add_argument("--slots",
                      action='store_true',
                      help="Whether to add slots info after node hostname or not")
  parser.set_defaults(slots=False)

  args = parser.parse_args()
  return args

if __name__ == '__main__':
  args = parse_args()
  
  output_nodefile_name = args.output_file
  gpus_per_node = args.num_gpus_per_node

  nodes_file = os.environ['PBS_NODEFILE']
  with open(nodes_file,"r") as f:
    nodes = f.readlines()
  with open(output_nodefile_name, 'w') as f:
    for node in nodes:
      if len(node.replace("\n","").replace(" ","")) != 0:
        f.write(node.replace("\n",""))
        f.write(" slots={}".format(gpus_per_node) if args.slots else "")
        f.write("\n")
    
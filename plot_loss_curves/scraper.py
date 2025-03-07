'''
Extract the loss and ppl from the a log file dumped from a training process.
The extracted info will be saved as a json file to the same path as the input log file.
'''

import argparse
from datetime import datetime
import os
import pandas as pd
import gc
import sys
import numpy as np


def parse_args():
  parser = argparse.ArgumentParser(description="Extract loss and ppl from log file")

  parser.add_argument('--log_files',
                      action='append',
                      required=True,
                      help='these log files will be scraped and results will be saved into one csv file')	
  parser.add_argument('--output_dir',
                        default='./results',
                        help='where the .csv file will be saved to')
  parser.add_argument('--mode',default='train',choices=['train','validation'],
                      help='whether scrape training loss or validation loss')
  parser.add_argument('--eval-interval', dest='eval_interval', default=100,
                      type=int,
                      help='how often is the evaluation performed')

  args = parser.parse_args()
  return args


def scrape_train_loss(args):
  iter, loss, ppl, time = [], [], [], []

  for log_file in args.log_files:
    with open(log_file,'r') as f:
      for line in f.readlines():
        line = line.replace("\n","")
        if "data-parallel-size" in line and "tensor-model-parallel-size" in line and "pipeline-model-parallel-size" in line:
          line_splitted = line.split(',')
          config = { item.split(":")[0].replace(" ",""):item.split(":")[1].replace(" ","") for item in line_splitted }
          data_parallel_size = config["data-parallel-size"]
          pipeline_parallel_size = config["pipeline-model-parallel-size"]
          tensor_paralle_size = config["tensor-model-parallel-size"]
        elif " iteration " in line and "consumed samples: " in line and "validation loss at iteration" not in line and 'checkpoint' not in line:
          line_splitted = line.split("|")
          
          # if 'nid' in line_splitted[0]:
          #   cur_iter = int(line_splitted[0].split('iteration')[1].replace(" ","").split("/")[0])
          # else:
          #   cur_iter = int(line_splitted[0].replace("iteration","").replace(" ","").split("/")[0])
          cur_iter = int(line_splitted[0].split('iteration')[1].replace(" ","").split("/")[0])

          iter.append(cur_iter)
          if "lm loss:" in line :
            idx = [i for i in range(len(line_splitted)) if "lm loss" in line_splitted[i]][0]
            cur_loss = float(line_splitted[idx].replace("lm loss:","").replace(" ",""))
            loss.append(cur_loss)
            ppl.append(np.exp(cur_loss))

            idx = [i for i in range(len(line_splitted)) if "elapsed time per iteration (ms)" in line_splitted[i]][0]
            time_cur_step = float(line_splitted[idx].replace(" elapsed time per iteration (ms):",'').replace(' ','')) / 1000 / 3600 / 24
            
            if (cur_iter - 1) % args.eval_interval == 0:
              time.append(time[-1] if len(time) > 0 else 0)
            else:
              time.append(time_cur_step)
          
        else:
          continue
  log_df = pd.DataFrame({"iteration":iter, "loss":loss, "ppl":ppl, "time":time}).drop_duplicates(subset='iteration',keep='first', ignore_index=True).sort_values(by='iteration')
  
  times = log_df['time'].tolist()
  for idx, time in enumerate(times):
    if idx > 0:
      times[idx] = times[idx] + times[idx-1]
  log_df['time'] = times
  log_df.to_csv(os.path.join(args.output_dir, f"log_train_PP{pipeline_parallel_size}_TP{tensor_paralle_size}_DP{data_parallel_size}.csv"), sep=",", index=False)


def scrape_val_loss(args):
  iter, losses, ppls = [], [], []
  for log_file in args.log_files:
    with open(log_file,'r') as f:
      for line in f.readlines():
        line = line.replace("\n","")
        if "data-parallel-size" in line and "tensor-model-parallel-size" in line and "pipeline-model-parallel-size" in line:
          line_splitted = line.split(',')
          config = { item.split(":")[0].replace(" ",""):item.split(":")[1].replace(" ","") for item in line_splitted }
          data_parallel_size = config["data-parallel-size"]
          pipeline_parallel_size = config["pipeline-model-parallel-size"]
          tensor_paralle_size = config["tensor-model-parallel-size"]
        elif " validation loss at iteration " in line:
          line_splitted = line.split("|")
          
          cur_iter = int(line_splitted[0].replace(" validation loss at iteration ","").replace(" ",""))
          iter.append(cur_iter)

          loss = float(line_splitted[1].replace(" lm loss value: ","").replace(" ",""))
          losses.append(loss)

          ppl = float(line_splitted[2].replace(" lm loss PPL: ","").replace(" ",""))
          ppls.append(ppl)
        else:
          continue
  log_df = pd.DataFrame({"iteration":iter, "loss":losses, "ppl":ppls}).drop_duplicates(subset='iteration',keep='first', ignore_index=True).sort_values(by='iteration')
  log_df.to_csv(os.path.join(args.output_dir, f"log_val_PP{pipeline_parallel_size}_TP{tensor_paralle_size}_DP{data_parallel_size}.csv"), sep=",", index=False)



if __name__ == '__main__':
  args = parse_args()
  
  if args.mode == 'train':
    scrape_train_loss(args)
  else:
    scrape_val_loss(args)

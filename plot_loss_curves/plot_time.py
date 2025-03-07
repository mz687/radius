import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import argparse
import matplotlib.font_manager as font_manager
import os
import numpy as np

title_font = {'fontname':'Sans serif',
              'fontsize':12,
              'weight':'bold'}
axis_font = {'fontname':'Sans serif',
             'fontsize':16}
legend_font = font_manager.FontProperties(family='Sans serif',
                                        style='normal', size=12)
ticks_fond = {'fontname':'Sans serif',
              'fontsize':10}

def parse_args():
  parser = argparse.ArgumentParser(description="Plot loss curve")

  parser.add_argument('--csv_files',
                        action='append',
                        help='where log files are',
                        required=True)	
  parser.add_argument('--output_dir',
                        default='./results',
                        help='where the figures will be saved to',
                        required=True)
  parser.add_argument('--model',default='GPT2 345M',type=str,
                      help='Model name and size',
                      required=True)
  parser.add_argument('--plot_interval',default=1000,type=int,
                      help='Interval between two x-axis ticks')
  parser.add_argument('--mode', default="train", type=str,
                      choices=['train', 'validation'],
                      help="Train or validation")
  parser.add_argument('--column_key', choices=['ppl', 'loss'], 
                      type=str, default='loss',
                      help='Plot either loss or ppl curve')
  parser.add_argument('--switch_iter', type=int, default=None,
                      help="Switch from dense to other reducer at which iteration")
  args = parser.parse_args()
  return args

def recompute_time(df_dict, switch_iter):
  baseline_df = None
  for key in df_dict.keys():
    if 'baseline' in key.lower():
      baseline_df = df_dict[key]
  assert baseline_df is not None

  for key, df in df_dict.items():
    if 'baseline' in key.lower():
      continue

    df['time'] = df['time'] + baseline_df['time'][switch_iter]
    df = pd.concat([baseline_df[:switch_iter], df], ignore_index=True)
    df_dict[key] = df

    # print(f"{key}[217588]: {df['time'][228493]} days")

    print(f"{key} speedup: {baseline_df['time'][len(df)-1] / df['time'][len(df)-1] :.4f}")
    
  return  df_dict

def plot_time(output_dir, df_dict, interval, model="GPT-345M", column_key='ppl', mode="train"):
  '''
  output_dir: abs path to the folder where png file(s) will be dumped
  df: pd.DataFrame, should have "iteration" and "loss" column.
  config: contains the training configurations: data parallel size, tensor parallel size, and pipeline parallel size
  '''
  xticks = None
  xticks_list=None
  largest_x_axis = float('-inf')
  for key, df in df_dict.items():
    start = int(df['iteration'].iloc[0])
    end = int(df['iteration'].iloc[-1])
    if largest_x_axis <= end:
      largest_x_axis = end
      xticks_list = [start, end]
  
  
  fig, ax = plt.subplots()
  y_max = -float('inf')
#   colors = ['#47855A', '#AB4F3F', '#1f77b4', '#E7BD39', '#384871']
  # colors = ['#E76254', '#528FAD', '#EF8A47', '#376795', '#F7AA58', '#72BCD5', '#1E466E']
  for idx, (key, df) in enumerate(df_dict.items()):
    print("Method: {}, time: {} days".format(key, df['time'][len(df)-1]))
    # ax.plot(df['time'],df[column_key],'-',color=colors[idx % len(colors)], linewidth=0.5,label=f"{key}")
    ax.plot(df['time'],df[column_key], '-', linewidth=0.5,label=f"{key}")
    y_max = max(y_max, df[column_key][0])
  # if xticks_list is not None and intrinsic_interval == 1:
  #   start,end = xticks_list
  #   xticks = [list(np.arange(start//100*100, end//100*100, interval, dtype=int)), list(np.arange(start//100*100/interval, end//100*100/interval, dtype=int))]
  # first = int(list(df_dict.values())[0]['iteration'].iloc[0])
  # second = int(list(df_dict.values())[0]['iteration'].iloc[1])
  # intrinsic_interval = second-first
  # if xticks is not None:
  #   ax.set_xticks(xticks[0],xticks[1])
  # ax.set_xlabel(f'Iteration (x{interval})',**axis_font)

  # plt.axvline(x = 1000, color = 'black', linewidth=1)

  # ax.get_xaxis().set_major_locator(matplotlib.ticker.MaxNLocator(nbins=(xticks_list[1]-xticks_list[0]) // interval))
  
  ax.set_xlabel(f'Number of Days',**axis_font)
  matplotlib.pyplot.autoscale(enable=True, axis='x')
  
  preset_y_max = 30
  if column_key == 'ppl':
    if y_max > preset_y_max:
      ax.set_ylim(10, preset_y_max)
      ax.set_yticks(list(range(10, preset_y_max+1, 5)), list(range(10, preset_y_max+1, 5)))
    if mode == 'train':
      ax.set_ylabel('LM Train Perplexity',**axis_font)
    else:
      ax.set_ylabel('LM Val. Perplexity',**axis_font)
  else:
    if mode == 'train':
      ax.set_ylabel('LM Train Loss',**axis_font)
    else:
      ax.set_ylabel('LM Val. Loss',**axis_font)
  # ax.set_title(f"Pre-train {model}", **title_font)
  ax.grid()
  ax.legend(prop=legend_font)
  fig.set_size_inches(5, 3)
  fig.tight_layout()
  fig.savefig(os.path.join(output_dir,f'plot_{column_key}_time.png'), dpi=400)
  plt.close()

if __name__ == '__main__':
  '''
  csv_file should have at least two columns: iteration and loss
  '''
  args = parse_args()
  csv_files = args.csv_files
  
  df_dict = {}
  
  print(csv_files)

  for csv_file in csv_files:
    csv_file_name_splitted = csv_file.split("/")[-1].replace(".csv","").split("_")[1:]
    comp_type=csv_file.split("/")[-2]
    DP,PP,TP = None, None, None

    for item in csv_file_name_splitted:
      if "DP" in item:
        DP = item.replace("DP","")
      elif "TP" in item:
        TP = item.replace("TP","")
      elif "PP" in item:
        PP = item.replace("PP","")
    # df = pd.read_csv(csv_file,sep=',').dropna()
    df = pd.read_csv(csv_file,sep=',').bfill()
    # key = "_".join([comp_type,f"PP{PP}", f"TP{TP}", f"DP{DP}"])
    if "baseline" in comp_type.lower() or "dense" in comp_type.lower():
      key = r"Baseline (Dense $allreduce$)"
    elif comp_type == 'lr_1.5e-4_density_0.01_range_50_interval_200' or comp_type == 'lr_1.5e-4_d_0.5_r_0_interval_200':
      key = r'Radius: $d$=0.5, $T$=200'
    elif comp_type == 'lr_1.5e-4_d_0.4_r_0_interval_400':
      key = r'Radius: $d$=0.4, $T$=200'
    elif comp_type == 'lr_1.5e-4_d_0.4_r_0_interval_200':
      key = r'Radius: $d$=0.4, $T$=400'
    else:
      key = comp_type
    df_dict[key] = df
  

  output_dir = args.output_dir

  if args.switch_iter:
    df_dict = recompute_time(df_dict, args.switch_iter)
  
  plot_time(output_dir, df_dict, args.plot_interval, args.model, args.column_key, mode=args.mode)
  
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

TRAIN_SAMPLING_PERIOD = 100
SAMPLING_PERIOD = 1
WINDOW_LENGTH, ORDER = 11, 1
EWM_ALPHA = 0.5
TRAIN_EWM_ALPHA = 0.2
ERROR_EWM_ALPHA = 0.5

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial
    
    try:
        window_size = np.abs(int(window_size))
        order = np.abs(int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')

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
                      help='Interval between two x-axis ticks',
                      required=True)
  parser.add_argument('--mode', default="train", type=str,
                      choices=['train', 'validation'],
                      help="Train or validation")
  parser.add_argument('--column_key', choices=['ppl', 'loss'], 
                      type=str, default='loss',
                      help='Plot either loss or ppl curve')
  args = parser.parse_args()
  return args

def plot(output_dir, df_dict, interval, model="GPT-345M", column_key='ppl', mode="train"):
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
  for key, df in df_dict.items():
    if "baseline" in key.lower() or 'dense' in key.lower():
      continue
    # ax.plot(df['iteration'],df[column_key],'-',linewidth=0.5,label=f"{key}")
    # y = df[column_key].to_numpy()[::SAMPLING_PERIOD]
    if mode == 'train':
      y = df[column_key][::TRAIN_SAMPLING_PERIOD].ewm(alpha=TRAIN_EWM_ALPHA, adjust=True).mean()
      x = df['iteration'].to_numpy()[::TRAIN_SAMPLING_PERIOD]
    else:
      y = df[column_key][::SAMPLING_PERIOD].ewm(alpha=EWM_ALPHA, adjust=True).mean()
      x = df['iteration'].to_numpy()[::SAMPLING_PERIOD]
    # ax.plot(x,y,'-',linewidth=0.5,label=f"{key}")
    ax.plot(x,y,'-',linewidth=1,label=f"{key}")
    y_max = max(y_max, df[column_key][0])
  
  # Always make sure baseline is on the top of the layers
  for key, df in df_dict.items():
    if "baseline" in key.lower() or 'dense' in key.lower():
      # ax.plot(df['iteration'],df[column_key],'-',linewidth=0.5,label=f"{key}")
      # y = savitzky_golay(df[column_key].to_numpy()[::SAMPLING_PERIOD], WINDOW_LENGTH, ORDER)
      # y = df[column_key].to_numpy()[::SAMPLING_PERIOD]
      if mode == 'train':
        y = df[column_key][::TRAIN_SAMPLING_PERIOD].ewm(alpha=TRAIN_EWM_ALPHA, adjust=True).mean()
        x = df['iteration'].to_numpy()[::TRAIN_SAMPLING_PERIOD]
      else:
        y = df[column_key][::SAMPLING_PERIOD].ewm(alpha=EWM_ALPHA, adjust=True).mean()
        x = df['iteration'].to_numpy()[::SAMPLING_PERIOD]
      # ax.plot(x,y,'-',linewidth=0.5,label=f"{key}")
      ax.plot(x,y,'-',linewidth=1,label=f"{key}")
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
  
  ax.set_xlabel(f'Step',**axis_font)
  matplotlib.pyplot.autoscale(enable=True, axis='x')
  if column_key == 'ppl':
    if y_max > 30:
      ax.set_ylim(10, 30)
      ax.set_yticks(list(range(10, 31, 5)), list(range(10, 31, 5)))
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

  if xticks_list is not None:
    start,end = xticks_list
    TARGET_INTERVAL = 50000
    INTERVAL = TARGET_INTERVAL // SAMPLING_PERIOD
    SCALE = 1000
    SCALED_INTERVAL = TARGET_INTERVAL // SCALE
    start = int(start // INTERVAL * INTERVAL) if start % INTERVAL != 0 else start
    # ax.set_xticks([x for x in range(start//1000*1000, end//1000*1000+1, INTERVAL)], [f'{x}k' for x in range(start//1000, end//1000+1, SCALED_INTERVAL)])
    if model == "GPT-355M":
      ax.set_xticks([50000, 100000, 150000, 200000, 250000, 300000, 350000, 400000, 450000, 500000], ['50k', '100k', '150k', '200k', '250k', '300k', '350k', '400k', '450k', '500k'])
    else:
      ax.set_xticks([50000, 100000, 150000, 200000, 250000, 300000], ['50k', '100k', '150k', '200k', '250k', '300k'])

  ax.grid()
  ax.legend(prop=legend_font)
  fig.set_size_inches(5, 3)
  fig.tight_layout()
  fig.savefig(os.path.join(output_dir,f'plot_{column_key}.png'), dpi=400)
  plt.close()


def plot_error(output_dir, df_dict, interval, model="GPT-355M", column_key='loss', abs_error=False, mode='train'):
  '''
  output_dir: abs path to the folder where png file(s) will be dumped
  df: pd.DataFrame, should have "iteration" and "loss" column.
  config: contains the training configurations: data parallel size, tensor parallel size, and pipeline parallel size
  '''
  assert len(df_dict) == 2
  xticks = None
  xticks_list=None
  smallest_x_axis = float('inf')
  freq = None
  for key, df in df_dict.items():
    if freq == None:
      freq = int(df['iteration'].iloc[1]) - int(df['iteration'].iloc[0])
    start = int(df['iteration'].iloc[0])
    end = int(df['iteration'].iloc[-1])
    if smallest_x_axis >= end:
      smallest_x_axis = end
      xticks_list = [start, end]
  
  
  key1, key2 = df_dict.keys()
  start, end = xticks_list
  if 'dense' in key1.lower() or 'baseline' in key1.lower():
    dense = df_dict[key1][column_key].to_numpy()[0:(end-start)//freq+1]
    topk = df_dict[key2][column_key].to_numpy()[0:(end-start)//freq+1]
  else:
    topk = df_dict[key1][column_key].to_numpy()[0:(end-start)//freq+1]
    dense = df_dict[key2][column_key].to_numpy()[0:(end-start)//freq+1]
  errors= topk - dense
  
  fig, ax = plt.subplots()

  if abs_error:
    errors = np.abs(errors)
  x_axis=list(np.arange(xticks_list[0], xticks_list[1]+1, freq, dtype=int))
  
  if not abs_error:
    pd.DataFrame(data=list(zip(x_axis, errors.tolist())), columns=['step','error']).to_csv(os.path.join(output_dir,'error.csv'), index=False, sep=',')
  
  errors = pd.Series(errors).ewm(alpha=ERROR_EWM_ALPHA, adjust=True).mean()
  # x = df['iteration'].to_numpy()[::SAMPLING_PERIOD]
  # ax.plot(x,y,'-',linewidth=0.5,label=f"{key}")
  ax.plot(x_axis,errors,'-',linewidth=0.5)
  
  # for key, df in df_dict.items():
  #   plt.plot(df['iteration'],df['loss'],'-',linewidth=0.5,label=f"{key}")
      
  if xticks_list is not None:
    start,end = xticks_list
    INTERVAL = 50000
    SCALE = 1000
    SCALED_INTERVAL = INTERVAL // SCALE
    start = int(start // INTERVAL * INTERVAL) if start % INTERVAL != 0 else start
    if model == "GPT-355M":
      ax.set_xticks([50000, 100000, 150000, 200000, 250000, 300000, 350000, 400000, 450000, 500000], ['50k', '100k', '150k', '200k', '250k', '300k', '350k', '400k', '450k', '500k'])
    else:
      ax.set_xticks([50000, 100000, 150000, 200000, 250000, 300000], ['50k', '100k', '150k', '200k', '250k', '300k'])

  ax.grid()
  # ax.legend(prop=legend_font)
  fig.set_size_inches(5, 3)
  
  if column_key == 'loss':
    ax.set_ylabel('Loss error',**axis_font)
    # if mode == 'train':
    #   ax.set_ylim(-0.025, 0.025)
  elif column_key == 'ppl':
    ax.set_ylabel('PPL error',**axis_font)

  ax.set_xlabel(f'Step',**axis_font)
  # plt.autoscale(enable=True, axis='x')
  fig.tight_layout()
  fig.savefig(os.path.join(output_dir,'plot_error.png' if not abs_error else 'plot_abs_error.png'), dpi=400)
  plt.close()



if __name__ == '__main__':
  '''
  csv_file should have at least two columns: iteration and loss
  '''
  args = parse_args()
  csv_files = args.csv_files
  
  df_dict = {}
  
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
    elif comp_type == 'lr_1.5e-4_d_0.1_r_0_interval_200':
      key = r'Radius: $d$=0.1, $T$=200'
    elif comp_type == 'lr_1.5e-4_d_0.9_r_0_interval_200':
      key = r'Radius: $d$=0.9, $T$=200'
    else:
      key = comp_type
    df_dict[key] = df
      
  output_dir = args.output_dir
  
  plot(output_dir, df_dict, args.plot_interval, args.model, args.column_key, mode=args.mode)
  
  if len(args.csv_files) == 2:
    plot_error(output_dir, df_dict, args.plot_interval, args.model, args.column_key, False, args.mode)

    plot_error(output_dir, df_dict, args.plot_interval, args.model, args.column_key, abs_error=True, mode=args.mode)

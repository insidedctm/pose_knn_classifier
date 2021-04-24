import argparse
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

def tsne(perp, iters, path = 'fitness_poses_csvs_out.csv'):
  ''' Run and plot TSNE
      PARAMS:
        perp - perplexity (typical value is between 5 and 50)
        iters - number of iterations to run TSNE 
  '''
  tsne, df, tsne_results = get_and_load_tsne(perp, iters, path)
 
  tsne_df = pd.DataFrame()
  tsne_df['tsne-2d-one'] = tsne_results[:,0]
  tsne_df['tsne-2d-two'] = tsne_results[:,1]
  tsne_df['exercise'] = df['exercise']
  exercise_name = df['exercise'].apply(lambda x: ' '.join(x.split('_')[:-1]))
  tsne_df['exercise_name'] = exercise_name
  up_down = df['exercise'].apply(lambda x: x.split('_')[-1])
  tsne_df['up_down'] = up_down
  source = df['filename'].apply(lambda x: x.split('.')[0])
  tsne_df['source'] = source
  
  print(set(exercise_name))
  print(set(up_down))
  print(set(source))
  print(tsne_df.shape)
  
  plt.figure(figsize=(16,10))
  sns.scatterplot(
      x="tsne-2d-one", y="tsne-2d-two",
      #hue="exercise_name",
      hue="source",
      style="up_down",
      #palette=sns.color_palette("Paired"),
      #palette=sns.color_palette("hls", n_exercises),
      data=tsne_df,
      legend="full",
      alpha=0.3
  )
  plt.show()

def get_and_load_tsne(perp, iters, path):
  data_cols = [f'data{ix}' for ix in range(99)]
  cols = ['filename', 'exercise']
  cols.extend(data_cols)
  df = pd.read_csv(path, names=cols)
  print(df)
  data_subset = df[data_cols].values
  
  tsne = TSNE(n_components=2, verbose=1, perplexity=perp, n_iter=iters)
  tsne_results = tsne.fit_transform(data_subset)
  
  return tsne, df, tsne_results
 
def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('perplexity', type=int)
  parser.add_argument('iterations', type=int)
  parser.add_argument('path', help='Path to CSV containing KNN samples')
  return parser.parse_args()

if __name__ == '__main__':
  args = parse_args()
  tsne(args.perplexity, args.iterations, args.path)

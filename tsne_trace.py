import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import argparse
from matplotlib.animation import FuncAnimation
from pose_embedder import FullBodyPoseEmbedder

def tsne_trace(train_path, perp, iter, plot_type, embedder):
  df, data_cols, data_subset = load_training_data(train_path)
  print(f'training data shape: {df.shape}')
  test_df = pd.DataFrame(load_test_data())
  test_df.columns = data_cols
  print(f'test data shape: {test_df.shape}')
  train_and_test_df = pd.concat([df[data_cols], test_df]).values
  if embedder:
    train_and_test_df = embed_points(train_and_test_df)
  print(f'train and test data shape: {train_and_test_df.shape}')
  tsne, tsne_results = run_tsne(train_and_test_df, perp, iter)
  print(tsne_results.shape)  

  tsne_df = pd.DataFrame()
  tsne_df['tsne-2d-one'] = tsne_results[:,0]
  tsne_df['tsne-2d-two'] = tsne_results[:,1]
  tsne_df['exercise'] = df['exercise']
  exercise_name = df['exercise'].apply(lambda x: ' '.join(x.split('_')[:-1]))
  exercise_name = exercise_name.append(pd.Series(['exercise'] * test_df.shape[0]), ignore_index = True)
  tsne_df['exercise_name'] = exercise_name
  up_down = df['exercise'].apply(lambda x: x.split('_')[-1])
  up_down = up_down.append(pd.Series(['up'] * test_df.shape[0]), ignore_index = True)
  tsne_df['up_down'] = up_down
  # source = df['filename'].apply(lambda x: x.split('.')[0])
  source = df['filename'].apply(lambda x: 'Train point')
  source = source.append(pd.Series(['Test point'] * test_df.shape[0]), ignore_index = True)
  tsne_df['source'] = source
  
  print(set(exercise_name))
  print(set(up_down))
  print(set(source))
  print(f'tsne data shape: {tsne_df.shape}')

  if plot_type == 'sns':
    sns_plot(tsne_df)
  elif plot_type == 'animate':
    animate_plot(tsne_df)
  else:
    print("unknown plot type, must be either 'animate' or 'sns'")

def embed_points(landmarks_coll):
  print(landmarks_coll)
  print(landmarks_coll.shape)
  embedder = FullBodyPoseEmbedder()
  result = [landmarks.reshape(33,3) for landmarks in landmarks_coll]
  result = np.array([embedder(landmarks) for landmarks in result])
  result = result.reshape(-1, 23*3)
  print(result)
  print(result.shape)
  return result
  

def animate_plot(df):
  nrows = len(df.index)
  test_len = np.sum(df['source'] == 'Test point')
  train_len = nrows - test_len
  print(f'[animate_plot] nrows={nrows}; train_len={train_len}; test_len={test_len}')

  fig = plt.figure(figsize=(16,10))
  train_df = df[:train_len]
  one = train_df['tsne-2d-one']
  two = train_df['tsne-2d-two']

  
  mask_ids = ['0', '1', 'begin', 'finish', '2']
  mask_ids.extend(['throughshoulders', 'standing', 'begin', 'allfours', 'up', 'crawling', 'finish', 'backonheels', 'plank', 'down'])
  colours  = ['g', 'b', 'y', 'm', 'c']
  colours.extend(['y','m','y','m','g','m','y','m','y','b'])
  for mask_id, colour in zip(mask_ids, colours): 
    mask = train_df['up_down'] == mask_id
    plt.scatter(one[mask], two[mask], c=colour)
  
  print(f"{train_df[mask]['tsne-2d-one']}")
  #plt.scatter(train_df[mask]['tsne-2d-one'], train_df[mask]['tsne-2d-two'], 'go')

  graph, = plt.plot([], [], 'r+-')

  def animate(i):
    print(f'animate with i={i}')
    print(f"  {df['tsne-2d-one'].iloc[train_len+i]}")
    graph.set_data(df['tsne-2d-one'][train_len:train_len+i], df['tsne-2d-two'][train_len:train_len+i])
    return graph

  ani = FuncAnimation(fig, animate, frames=test_len, interval=200)

  # set plot limits
  one = df['tsne-2d-one']
  two = df['tsne-2d-two']
  plt.xlim([one.min(), one.max()])
  plt.ylim([two.min(), two.max()])
 
  plt.show()

def sns_plot(tsne_df):  
  plt.figure(figsize=(16,10))
  sns.scatterplot(
      x="tsne-2d-one", y="tsne-2d-two",
      #hue="exercise_name",
      hue="up_down",
      style="source",
      #palette=sns.color_palette("Paired"),
      #palette=sns.color_palette("hls", n_exercises),
      data=tsne_df,
      legend="full",
      alpha=0.3
  )
  plt.show()

def load_training_data(path):
  data_cols = [f'data{ix}' for ix in range(99)]
  cols = ['filename', 'exercise']
  cols.extend(data_cols)
  df = pd.read_csv(path, names=cols)
  print(df)
  data_subset = df[data_cols].values
  return df, data_cols, data_subset


def run_tsne(data_subset, perp, iters):  
  tsne = TSNE(n_components=2, verbose=1, perplexity=perp, n_iter=iters)
  tsne_results = tsne.fit_transform(data_subset)
  
  return tsne, tsne_results


def load_test_data():
  test_points = np.loadtxt('temp.csv', delimiter=',')
  return test_points

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('train_path', help='Path to CSV of training samples')
  parser.add_argument('--perp', default=50, help='TSNE perplexity to use')
  parser.add_argument('--iter', default=500, help='TSNE iterations to use')
  parser.add_argument('--plot_type', default='animate', help='Type of plot [animate|sns]')

  # embedder/no-embedder
  parser.add_argument('--embedder', dest='embedder', action='store_true')
  parser.add_argument('--no-embedder', dest='embedder', action='store_false')
  parser.set_defaults(embedder=True)

  return parser.parse_args()

args = parse_args()
tsne_trace(args.train_path, int(args.perp), int(args.iter), args.plot_type, args.embedder)

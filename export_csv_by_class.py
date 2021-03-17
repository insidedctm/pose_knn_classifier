import os
import pandas as pd

def export_csv_by_class(input='fitness_poses_csvs_out.csv', output_folder='csvs'):
  df = pd.read_csv(input, header=None)

  exercise_to_class_map = get_exercise_to_class_map(df)
  print(exercise_to_class_map)

  if not os.path.exists(output_folder):
    os.makedirs(output_folder)

  for ex, classes in exercise_to_class_map.items():
    output_csv(df, ex, classes, output_folder)

def output_csv(df, exercise, classes, output_folder):
  output_path = f'{output_folder}/{exercise}.csv'
  print(f'output csv for {exercise} to {output_path}')
  print(f'  classes={classes}')

  exercise_df = df[df[1].isin(classes)]
  exercise_df.to_csv(output_path, index=None, header=None)
  

def get_exercise_to_class_map(df):
  ''' Get list of exercises based on the class names given in the data frame (df)
    Class names are lower-case identifiers for an exercise pose with words
    separated by underscores. The format is <exercise>_<pose> where <exercise>
    can be underscore-separated but <pose> is alphanumeric with no underscore.
    Therefore the class name can be separated by splitting on the final underscore.
    
      The return value is a dictionary where <exercise> is the key and the value
    is a list containing the exercise poses.     
  '''

  exercise_poses = set(df[1])

  exercises = set(['_'.join(ex.split('_')[:-1]) for ex in exercise_poses])

  get_poses = lambda ex: [expose for expose in exercise_poses if expose.startswith(ex)]
  map = {ex:get_poses(ex) for ex in exercises}

  return map


export_csv_by_class()

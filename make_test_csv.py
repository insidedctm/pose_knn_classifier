import argparse
import glob
import pandas as pd

def make_test_csv(test_data_dir, csv_path):
  print(f'making test csv from {test_data_dir}')

  image_dirs = glob.glob(f'{test_data_dir}/*')

  dir_class = [dir.split('/') for dir in image_dirs]
  dir_class = [(el[0], el[1].split('_')) for el in dir_class]
  dir_class = [(el[0], el[1][:-1], el[1][-1]) for el in dir_class]
  dir_class = [( f"{el[0]}/{'_'.join(el[1])}", '_'.join(el[1]), el[2] ) for el in dir_class]  
  print(dir_class)

  filenames = []
  classifications = []
  exercises = []
  for dir, exercise, classification in dir_class:
    imagepaths = glob.glob(f'{dir}_{classification}/*')
    filenames.extend(imagepaths)
    classifications.extend([classification] * len(imagepaths))
    exercises.extend([exercise] * len(imagepaths))

  df = pd.DataFrame({
           'filepaths': filenames,
           'exercise': exercises,
           'groundtruth': classifications
  })  
  df.to_csv(csv_path, header=True, index=None)
  
def parse_args():
  description = '''
    Processes subfolders in input directory to record filepath and ground-truth classification
    for each image file. Folder structure:
          input_dir/
             <exercise>_<classification1>/
                filename.jpg
                ...
             <exercise>_<classification2>/
                ...
  '''
  parser = argparse.ArgumentParser(description=description)
  parser.add_argument('data_dir')
  parser.add_argument('csv_path')
  return parser.parse_args()

if __name__ == '__main__':
  args = parse_args()
  print(args)
    
  make_test_csv(args.data_dir, args.csv_path)

import argparse
import pandas as pd
import glob

def export_csv(input_dir, out_name, non_terminal_class_list):
  all_files = glob.glob(input_dir + "/*.csv")
  li = []
  for filename in all_files:
    class_name = filename.split('.csv')[0].split('_')[-1]
    print(f'{filename} -> {class_name}')
    df = pd.read_csv(filename, index_col=None, header=0)
    data_cols = [f'data{ix}' for ix in range(1,100)]
    print(f'data_cols={data_cols}')
    print(df.columns)
    df.columns = ['filename'] + data_cols
    print(f'df.columns={df.columns}')
    df['class_name'] = class_name
    li.append(df)

  frame = pd.concat(li, axis=0, ignore_index=True)
  print(frame)
  print(frame.groupby('class_name').count())

  mapping = get_class_name_mapping(non_terminal_class_list)

  frame['class_name'] = [mapping[cn] for cn in frame['class_name']]

  frame = frame[['filename', 'class_name'] + data_cols]

  frame.to_csv(out_name, header=None, index=None)

def get_class_name_mapping(non_terminals):
  mapping = {'begin': 'begin'}
  for ix, non_terminal in enumerate(non_terminals.split(',')):
    mapping[non_terminal] = ix
  mapping['finish'] = 'finish'
  print(mapping)
  return mapping
    


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('input_dir')
  parser.add_argument('out_name')
  parser.add_argument('non_terminal_class_list')
  return parser.parse_args()  

if __name__ == '__main__':
  args = parse_args()
  print(args)

  export_csv(args.input_dir, args.out_name, args.non_terminal_class_list)
  

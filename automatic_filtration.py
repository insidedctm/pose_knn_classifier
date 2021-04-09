import argparse
from bootstrap_helper import BootstrapHelper
from pose_embedder import FullBodyPoseEmbedder
from pose_classifier import PoseClassifier

# Required structure of the images_in_folder:
#
#   fitness_poses_images_in/
#     pushups_up/
#       image_001.jpg
#       image_002.jpg
#       ...
#     pushups_down/
#       image_001.jpg
#       image_002.jpg
#       ...
#     ...
BOOTSTRAP_IMAGES_IN_FOLDER_DEFAULT = 'fitness_poses_images_in'

# Output folders for bootstrapped images and CSVs.
bootstrap_images_out_folder = 'fitness_poses_images_out'
BOOTSTRAP_CSVS_OUT_FOLDER_DEFAULT = 'fitness_poses_csvs_out'

def filtration(bootstrap_images_in_folder, bootstrap_csvs_out_folder):
  # Initialize helper.
  bootstrap_helper = BootstrapHelper(
      images_in_folder=bootstrap_images_in_folder,
      images_out_folder=bootstrap_images_out_folder,
      csvs_out_folder=bootstrap_csvs_out_folder,
  )
  

  # Find outliers.
  
  # Transforms pose landmarks into embedding.
  pose_embedder = FullBodyPoseEmbedder()
  
  # Classifies give pose against database of poses.
  pose_classifier = PoseClassifier(
      pose_samples_folder=bootstrap_csvs_out_folder,
      pose_embedder=pose_embedder,
      top_n_by_max_distance=30,
      top_n_by_mean_distance=10)
  
  outliers = pose_classifier.find_pose_sample_outliers()
  print('Number of outliers: ', len(outliers))
  
  bootstrap_helper.analyze_outliers(outliers)
  
  # Remove all outliers (if you don't want to manually pick).
  bootstrap_helper.remove_outliers(outliers)
  
  # Align CSVs with images after removing outliers.
  bootstrap_helper.align_images_and_csvs(print_removed_items=False)
  bootstrap_helper.print_images_out_statistics()

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument(	'--bootstrap_in', 
			default=BOOTSTRAP_IMAGES_IN_FOLDER_DEFAULT,
			help='')
  parser.add_argument(	'--csvs_out',
			default=BOOTSTRAP_CSVS_OUT_FOLDER_DEFAULT,
			help='')
  return parser.parse_args()

if __name__ == '__main__':
  args = parse_args()
  
  print(args)

  filtration(args.bootstrap_in, args.csvs_out)

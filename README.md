# Human Pose K-Nearest Neighbour Classifier

## Prerequisites
python 3.7 installation

## Installation
```{bash}
git clone ...
cd pose_knn_classifier
pip install -r requirements.txt

## Operation
1. Record videos of exercises
2. Extract frames of exercise up and down position 
3. Run bootstrapping, filtration and csv export

The above steps produces a single csv file containing all the data required for the KNN classifier

### Extract frames
```{bash}
python extract_frames.py <video> fitness_poses_images_in/ <exercise>
```

'''{bash}
python extract_frames.py y-squats.mp4 fitness_poses_images_in/ y_squats
```

Running this python program will provide an on-screen interface to step through the video and output frames to the up or down folder.

The up folder will be located in the path `fitness_poses_images_in/<exercise>_up`. Similarly
the down folder will be located in the path `fitness_poses_images_in/<exercise>_down`.

### Bootstrapping
```{bash}
python bootstrap.py
```

This program will use the images in fitness_poses_images_in to write out csv files for each exercise up/down class in `fitness_poses_csvs_out`.

### Filtration
```{bash}
python automatic_filtration.py
```

This program uses the KNN classifier to score each of the poses and detects any outlier poses. These poses are then removed from the samples to be 
used by the classifier.

### CSV export
```{bash}
python export_csv.py
```

Combines all the filtered csv into a single csv called `fitness_poses_csvs_out.csv`. This file can be imported into the Android app to provide
the comparison data for the KNN classifier.

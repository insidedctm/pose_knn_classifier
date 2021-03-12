# Human Pose K-Nearest Neighbour Classifier

This repo contains code to manage the extraction of pose samples from recorded videos
to provide the training corpus for the Nearest Neighbour classifier (which is used to implement
exercise rep counting). The output of the programs contained here is a CSV file containing
pose samples that can be loaded into the MLKit Pose Classifier.

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

## Analsysis of pose samples
To get an idea of how well clustered samples for a class compared to other classes we can run T-distributed Stochastic Neighbour Embedding (T-SNE)
on the final CSV file. T-SNE finds a 2-d embedding of the 99-dimensional pose samples so that they can be conveniently plotted.

```{bash}
python tsne.py <perplexity> <iterations>
```

Perplexity is usually between 5 and 100; 50 seems to give good results. Iterations should be set high enough so that the embedding converges to 
something visually useful. This can be found empirically however 500 has given good results.

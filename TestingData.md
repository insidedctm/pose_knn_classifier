# Creating Test Data

Tim has created videos of various exercises showing good examples of the exercise followed by an
exercise containing a specific error. These videos has been uploaded to S3 in the `stonefish-data/correctional-points-example-videos/`.

We can use these videos as sources of test data for both rep counting and correctional point detection. The script
`extract_testing_data.py` preprocesses the videos, extracting the pose from each frame and saving to a single CSV file.

The following command creates the file `output.csv`

```{bash}
python extract_testing_data.py stonefish-data correctional-points-example-videos
```

## File format

The file contains one row for each frame where a pose is detected. The format is

```{bash}
<exercise - correction point>, 123.2, 310.0, -120.2, ..., 87.9, 258.6, -109.2
                               <- x,y,z for lm 1 ->       <- x,y,x for lm 33->
```



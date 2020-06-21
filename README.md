# asl_handgesture_classification
classification of ASL (American Sign Language) hand gestures

## Project description

The project performs Machine Learning based ASL Fingerspelling classification from segmented hand gesture images.

![asl handgesture 'A'](https://github.com/Mekhak/asl_handgesture_classification/blob/master/data/splited/test/a/a.13.png)
![asl handgesture 'J'](https://github.com/Mekhak/asl_handgesture_classification/blob/master/data/splited/test/j/j.22.png)
![asl handgesture 'C'](https://github.com/Mekhak/asl_handgesture_classification/blob/master/data/splited/test/c/c.1.png)


## Project structure

The project is splited into Jupyter Notebook files:

- *train_logistic_regression.ipynb*:  trains logistic regression model.
- *train_random_forest.ipynb*:  trains random forest model.
- *train_cnn.ipynb*: trains convolutional neural net with (custum architecture).
- *VGG16_fine_tuned.ipynb*: fine tunes `VGG16` model (transfer learning).
- *predict.py*:  predict on user specified image. Run `python predict.py --image imagepath`
- *split_data_into_seperate_folders.ipynb*:  splits images into train, validation and test parts (this part is not required to run and reproduce the results).
- *convert_images_to_csv.ipynb*:  converts images to `pandas` `DataFrame` for easy manipulation with `scikit-learn` (this part is not required to run and reproduce the results).

## Results
The following results were achived durring the training:

- **Logistic Regression:**  98% test accuracy
- **Random Forest:**  95% test accuracy
- **CNN:** 94% test accuracy
- **VGG16 (transfer learning):** 98% test accuracy

## Dependencies
The dependencies are listed in `environment.yml` file.
To create `Anaconda` envirenment with all the required packages installed: `conda env create -f environment.yml`.

Note that the convolutional neural nets are trained on GPU. To replicate those training parts install the GPU version of `Tensorflow` (training on CPU will last long time).

## Future work
Planning to integrate an image segmentation functionality and wrap up the project as a complete working product on non segmented real world images.

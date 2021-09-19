Automated Essay Scoring
=======================

## Table of contents
* [Introduction](#introduction)
* [Dataset](#dataset)
* [Kaggle Placement](#kaggle-results)
* [Results](#results)
* [Libraries](#libraries)

## Introduction
Preprocessed data derived from a set of essays (feature selection, normalization, oversampling), then conducted predictive analysis utilizing machine learning models such as SVMs and ensemble classifiers such as Random Forest Classifiers. At the end, the various models' accuracies are evaluated with the help of a confusion matrix and a quadratic kappa score.

## Dataset
The dataset used for this project was obtained from a [popular Kaggle competition](https://www.kaggle.com/c/asap-aes) sponsored by the Hewlett Foundation.

## Kaggle Results
With the model built from the Random Forest Classifier, I **won the Kaggle competition with 172 participating teams**.
[Full leaderboard](https://www.kaggle.com/c/mum-fit1043-s1-2021/leaderboard)
![Leaderboard](./Results/kaggle_results.JPG)

## Results
### SVM model (Confusion Matrix)
![SVM Confusion Matrix](./Results/svm_confusion_matrix.JPG)
### SVM model (Quadratic Kappa Score)
![SVM Quadratic Kappa Score](./Results/svm_qwk.JPG)
### RFC model (Confusion Matrix)
Note: Performed oversampling, thus total number of samples is greater in comparison to the SVM Confusion Matrix, the goal here is to correctly classify infrequent scores (1 and 6)

![RFC Confusion Matrix](./Results/rfc_confusion_matrix.JPG)
### RFC model (Quadratic Kappa Score)
Note: Much better QWK score in comparison to the SVM model!
![RFC Quadratic Kappa Score](./Results/rfc_qwk.JPG)


## Libraries
* scikit-learn (sklearn)
* imbalanced-learn (imblearn)
* pandas
* numpy

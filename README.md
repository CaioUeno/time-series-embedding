# Time Series Embedding

This repository contains experiments of using time series embeddings learnt by a neural network.

## Table of contents

* [Methodology](#Methodology)
* [Dependencies](#Dependencies)
* [Report](#Report)
    * [Nearest Neighbour](#(One)-Nearest-Neighbour)

## Methodology

## Dependencies

* numpy
* sklearn
* pandas
* tensorflow
* [dtw-python](https://github.com/DynamicTimeWarping/dtw-python)

## Report

This section shows the comparison reports of some datasets - download Full-Report.csv file for all results.

### (One) Nearest Neighbour

|            Dataset            | Train Silhouette Improvement | Test Silhouette Improvement | Euclidean Distance |   DTW   | Embedded + Cosine Distance |
| :---------------------------: | :--------------------------: | :-------------------------: | :----------------: | :-----: | :------------------------: |
| FreezerRegularTrain           | +0.8095                      | +0.7392                     | .8049              | .873    | **.982**                   |
| ProximalPhalanxOutlineCorrect | +0.7364                      | +0.5753                     | .8076              | .7526   | **.8625**                  |
| PowerCons                     | +0.6375                      | +0.6168                     | .9778              | .8778   | **.9852**                  |

### Three Nearest Neighbours

|            Dataset            | Train Silhouette Improvement | Test Silhouette Improvement | Euclidean Distance |   DTW   | Embedded + Cosine Distance |
| :---------------------------: | :--------------------------: | :-------------------------: | :----------------: | :-----: | :------------------------: |
| FreezerRegularTrain           | +0.                      | +0.                     | .              | .    | **.**                   |
| ProximalPhalanxOutlineCorrect | +0.                      | +0.                     | .              | .   | **.**                  |
| PowerCons                     | +0.                      | +0.                     | .              | .   | **.**                  |

### Best Of

|            Dataset            | Train Silhouette Improvement | Test Silhouette Improvement | Euclidean Distance |   DTW   | Embedded + Cosine Distance |
| :---------------------------: | :--------------------------: | :-------------------------: | :----------------: | :-----: | :------------------------: |
| FreezerRegularTrain           | +0.                      | +0.                     | .              | .    | **.**                   |
| ProximalPhalanxOutlineCorrect | +0.                      | +0.                     | .              | .   | **.**                  |
| PowerCons                     | +0.                      | +0.                     | .              | .   | **.**                  |

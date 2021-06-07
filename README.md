# Time Series Embedding

This repository contains experiments of embedding time series using a neural network.

## Table of contents

* [Methodology](#Methodology)
* [Dependencies](#Dependencies)
* [Report](#Report)
    * [Nearest Neighbour](#(One)-Nearest-Neighbour)

## Methodology

The concept is to embed time series and build a new space of instances. The instances' class is the only information that drives the modeling. There are *two* neural network models - embedder and trainable model - and the embedder is **inside** the trainable model as show in the figure.

![Architecture](https://github.com/CaioUeno/time-series-embedding/blob/master/images/Architecture.png)

The training is as follow: given a pair of instances, the trainable model embeds them using the embedder model and outputs the normalized intern product between their embeddings (**cosine similarity**). There are possible labels: if they belong to the same class then the label is **1**, otherwise it is **-1**. These values mean that their embeddings should be in the **same** or **opposite** directions, respectively. Notice that it is applied (and works) only on binary classification datasets.

After the training step, we use only the embedder model to embed the time series. Finally, we apply a distance based machine learning algorithm to classify instances.

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
| Strawberry                    | +0.7992                      | +0.7532                     | .9459              | .9378   | **.9595**                  |
| SonyAIBORobotSurface1         | +0.6994                      | +0.2413                     | .6955              | .6639   | **.8247**                  |

### Three Nearest Neighbours

|            Dataset            | Train Silhouette Improvement | Test Silhouette Improvement | Euclidean Distance |   DTW   | Embedded + Cosine Distance |
| :---------------------------: | :--------------------------: | :-------------------------: | :----------------: | :-----: | :------------------------: |
| FreezerRegularTrain           | +0.8095                      | +0.7392                     | .7842              | .8179   | **.982**                   |
| ProximalPhalanxOutlineCorrect | +0.7364                      | +0.5753                     | .8488              | .7698   | **.8855**                  |
| PowerCons                     | +0.6375                      | +0.6168                     | .9667              | .8611   | **.9741**                  |
| Strawberry                    | +0.7992                      | +0.7532                     | .9243              | .9297   | **.9649**                  |
| SonyAIBORobotSurface1         | +0.6994                      | +0.2413                     | .574               | .5624   | **.8292**                  |

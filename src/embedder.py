import pandas as pd
import numpy as np

from tensorflow.keras.layers import Input, Dense, GRU, Bidirectional, BatchNormalization, Dot, Reshape
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.math import scalar_mul, multiply, negative, log, add, subtract, maximum
from tensorflow import constant, shape
from tensorflow.keras.models import Model


class TimeSeriesEmbedder(object):

    """
    Embedder neural network model for time series.

    Arguments:

    """

    def __init__(
        self, embedding_length: int, time_series_length: int, n_features: int = 1, recurrent_units: int = 32, base_embedder = None
    ):

        # attributes
        self.embedding_length = embedding_length
        self.time_series_length = time_series_length
        self.n_features = n_features
        self.recurrent_units = recurrent_units

        # embedder model was passed 
        if base_embedder:
            self.base_embedder = base_embedder
        else:
            self.base_embedder = self.get_standard_model()

        # trainable model
        self.trainable_model = self.get_trainable_model()

    def get_standard_model(self):

        """
        Auxiliary method to build the embedder model.

        Returns:
            base_embedder: model which can generate the embedding of a given time series.
        """

        # input layer
        input_layer = Input(shape=((self.time_series_length, self.n_features)), name="input_layer")

        # recurrent layer
        recurrent_layer = GRU(self.recurrent_units, name="recurrent_layer")
        # bidirectional layer
        gru_layer = Bidirectional(recurrent_layer, name="bidirectional_layer")(
            input_layer
        )
        # normalization layer
        norm_layer =  BatchNormalization()(gru_layer)

        # final layer - embedding
        embedding = Dense(self.embedding_length)(norm_layer)

        return Model(inputs=input_layer, outputs=embedding)

    def get_trainable_model(self):

        """
        Auxiliary method to build the trainable model - generates embeddings for two instances and computes their cosine similarity.

        Returns:
            trainable_model: model that outputs the cosine similarity between two time series' embeddings.
        """

        # input for two instances
        instance_one = Input(shape=(self.time_series_length, self.n_features), name="instance_one")
        instance_two = Input(shape=(self.time_series_length, self.n_features), name="instance_two")

        # dot product simulates a intern product between vectors
        # also, adding normalize=True, it returns the **cosine similarity** between them
        dot_layer = Dot(axes=1, normalize=True)(
            [self.base_embedder(instance_one), self.base_embedder(instance_two)]
        )

        trainable_model = Model(
            inputs=[
                instance_one,
                instance_two,
            ],
            outputs=dot_layer,
        )

        trainable_model.compile(loss="mae", optimizer="adam")

        return trainable_model

    def fit(
        self,
        X,
        y,
        epochs: int = 25,
        batch_size: int = 128,
    ):

        """
        Method to fit the trainable model.

        Arguments:
            X (np.array): time series data - with shape (number of time series, time series length, number of features per timestep).
            y (np.array): time series' labels.
            batch_size (int): batch size.
            epochs (int): how many epochs to fit the model.
        """

        def instances_generator(X, y, batch_size):

            """
            Generator function to generate instance pair and label: (ts, ts) - similarity.
            """

            while True:

                # use pandas cartesian product to create training set
                # df has two columns: index and class
                df = (
                    pd.DataFrame(y)
                    .rename(columns={0: "class"})
                    .reset_index()
                    .sample(batch_size, replace=False)
                )
                df = df.merge(df, how="cross")

                # if two time series have the same class their cosine similarity is 1 and 0 otherwise
                df["label"] = (df["class_x"] == df["class_y"]).astype(float)

                # binary classification - forcing classes to be opposites
                if len(np.unique(y)) == 2:
                    df["label"].map({1: 1, 0: -1})

                # using index column to get instances
                X_1 = X[df["index_x"]]
                X_2 = X[df["index_y"]]

                yield [X_1, X_2], np.array(df["label"])

        gen = instances_generator(X, y, batch_size)

        steps_per_epoch = (X.shape[0] ** 2) // (batch_size ** 2)

        self.trainable_model.fit(
            gen, epochs=epochs, steps_per_epoch=steps_per_epoch
        )

    def encode(self, X):

        """
        Method to encode time series.

        Arguments:
            X (np.array): time series data - with shape (number of time series, time series length, number of features per timestep).
        """
        
        return self.base_embedder.predict(X)

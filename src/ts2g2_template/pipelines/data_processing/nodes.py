from typing import Dict, Tuple

import pandas as pd
from ts2g2 import Timeseries, TimeseriesPreprocessing, TimeseriesPreprocessingSegmentation, TimeseriesPreprocessingSlidingWindow, TimeseriesPreprocessingComposite, TimeseriesView, TimeGraph, ToSequenceVisitorSlidingWindow, ToSequenceVisitor, ToSequenceVisitorOrdinalPartition

import uuid
from ts2g2 import BuildTimeseriesToGraphNaturalVisibilityStrategy, BuildTimeseriesToGraphHorizontalVisibilityStrategy, BuildTimeseriesToGraphOrdinalPartition, BuildTimeseriesToGraphQuantile
from ts2g2 import StrategyLinkingGraphByValueWithinRange, LinkNodesWithinGraph
from ts2g2 import get_random_walks_from_graph_df
from ts2g2 import train_graph_embedding_model_df
# from ts2vec import TS2Vec
#from src.ts2vec.ts2vec import TS2Vec
import networkx as nx
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import numpy as np


# Function to add UUIDs to the DataFrame
def add_uuid_column(amazon: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a UUID column to the Amazon DataFrame.

    Args:
        amazon (pd.DataFrame): DataFrame containing Amazon stock data.
    
    Returns:
        pd.DataFrame: Updated DataFrame with an additional 'UUID' column.
    """
    amazon['UUID'] = [uuid.uuid4() for _ in range(len(amazon))]

    return amazon


# Function to add a column of vectors for the last 10 closing values
def add_vector_column(amazon: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """
    Adds a new column of vectors for the last n closing values.

    Args:
        amazon (pd.DataFrame): DataFrame containing Amazon stock data.
        n (int): Number of previous closing values to include in the vector.
    
    Returns:
        pd.DataFrame: Updated DataFrame with a new column of vectors.
    """
    amazon['Last_Close_Values'] = [window.to_list() for window in amazon['Close'].rolling(window=n)]
    return amazon



# Visibility graph node
def create_visibility_graph(amazon: pd.DataFrame):
    """
    Adds new column to a dataframe containing natural visibility graph from time series data of vectors.
    
    Args:
        amazon (pd.DataFrame): DataFrame containing time series data.
    
    Returns:
        pd.DataFrame: Dataframe containing new column with graphs
    """
    graphs = []
    for vec in amazon['Last_Close_Values']:
        graph = Timeseries(vec).with_preprocessing(TimeseriesPreprocessing())\
        .to_graph(BuildTimeseriesToGraphNaturalVisibilityStrategy().with_limit(1).get_strategy())\
        .link(LinkNodesWithinGraph().by_value(StrategyLinkingGraphByValueWithinRange(2)).seasonalities(15))

        graphs.append(graph)

    amazon['Graph'] = graphs

    return amazon


# Ordinal partition graph node
def create_ordinal_partition_graph(amazon: pd.DataFrame):
    """
    Adds new column to a dataframe containing ordinal partition graph from time series data of vectors.
    
    Args:
        amazon (pd.DataFrame): DataFrame containing time series data.
    
    Returns:
        pd.DataFrame: Dataframe containing new column with graphs
    """
    graphs = []
    for vec in amazon['Last_Close_Values']:
        graph = Timeseries(vec).with_preprocessing(TimeseriesPreprocessing())\
        .to_graph(BuildTimeseriesToGraphOrdinalPartition(3, 2).get_strategy())

        graphs.append(graph)

    amazon['Graph'] = graphs

    return amazon


# Ordinal partition graph node
def create_quantile_graph(amazon: pd.DataFrame):
    """
    Adds new column to a dataframe containing quantile graph from time series data of vectors.
    
    Args:
        amazon (pd.DataFrame): DataFrame containing time series data.
    
    Returns:
        pd.DataFrame: Dataframe containing new column with graphs
    """
    graphs = []
    for vec in amazon['Last_Close_Values']:
        graph = Timeseries(pd.Series(vec)).with_preprocessing(TimeseriesPreprocessing())\
        .to_graph(BuildTimeseriesToGraphQuantile(4, 1).get_strategy())

        graphs.append(graph)

    amazon['Graph'] = graphs

    return amazon


# Time series embedding
""" def ts_embedding(amazon: pd.DataFrame):

    time_series_data = amazon['Close'] 

    model = ts2vec(input_dims=1) """


# Get random walks
def get_random_walks_from_graphs(amazon: pd.DataFrame) -> pd.DataFrame:

    amazon_rand_walk = get_random_walks_from_graph_df(amazon)
    
    return amazon_rand_walk



def train_graph_embedding_model(amazon_rand_walk: pd.DataFrame, embedding_size: int = 20):
    """
    Trains a Doc2Vec model on random walks generated for each row in the DataFrame.

    This function takes the random walks in the 'Random_Walks' column for each row,
    transforms these walks into TaggedDocument format for Doc2Vec, and trains a Doc2Vec 
    model with the specified embedding size. The trained model is then stored in the 
    'Doc2Vec_Model' column for each row.

    Args:
        amazon_rand_walk (pd.DataFrame): DataFrame with a column 'Random_Walks' containing
                                          random walks.
        embedding_size (int): Size of the embeddings to be trained.

    Returns:
        pd.DataFrame: DataFrame with an additional column 'Doc2Vec_Model' containing the
                      trained Doc2Vec models for each row.
    """
    amazon_model = train_graph_embedding_model_df(amazon_rand_walk, embedding_size)

            
    return amazon_model












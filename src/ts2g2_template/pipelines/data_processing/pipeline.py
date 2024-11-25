from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    add_uuid_column,
    add_vector_column,
    create_visibility_graph,
    create_ordinal_partition_graph,
    create_quantile_graph,
    get_random_walks_from_graphs,
    train_graph_embedding_model

)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=add_uuid_column,
                inputs="amazon_data",
                outputs="amazon_data_with_uuid",
                name="add_uuid_column_node",
            ),
            node(
                func=add_vector_column,
                inputs="amazon_data_with_uuid",
                outputs="amazon_data_with_vectors",
                name="add_vector_column_node",
            ),
            node(
                func=create_visibility_graph,
                inputs="amazon_data_with_vectors",
                outputs="amazon_data_with_visibility_graph",
                name="create_visibility_graph_node",
            ),
            node(
                func=create_ordinal_partition_graph,
                inputs="amazon_data_with_vectors",
                outputs="amazon_data_with_ordinal_partition_graph",
                name="create_ordinal_partition_graph_node",
            ),
            node(
                func=create_quantile_graph,
                inputs="amazon_data_with_vectors",
                outputs="amazon_data_with_quantile_graph",
                name="create_quantile_graph_node",
            ),
            node(func=get_random_walks_from_graphs, 
                inputs="amazon_data_with_visibility_graph", 
                outputs="amazon_data_with_rand_walks_visibility", 
                name="apply_random_walks_visibility_graph_node"
            ),
            node(func=get_random_walks_from_graphs, 
                inputs="amazon_data_with_ordinal_partition_graph", 
                outputs="amazon_data_with_rand_walks_ordinal_partition", 
                name="apply_random_walks_ordinal_partition_graph_node"
            ),
            node(func=get_random_walks_from_graphs, 
                inputs="amazon_data_with_quantile_graph", 
                outputs="amazon_data_with_rand_walks_quantile", 
                name="apply_random_walks_quantile_graph_node"
            ),
            node(func=train_graph_embedding_model, 
                inputs="amazon_data_with_rand_walks_visibility", 
                outputs="visibility_graph_embedding_model", 
                name="train_visibility_graph_embedding_model_node"
            ),
            node(func=train_graph_embedding_model, 
                inputs="amazon_data_with_rand_walks_ordinal_partition", 
                outputs="ordinal_partition_graph_embedding_model", 
                name="train_ordinal_partition_graph_embedding_model_node"
            ),
            node(func=train_graph_embedding_model, 
                inputs="amazon_data_with_rand_walks_quantile", 
                outputs="quantile_graph_embedding_model", 
                name="train_quantile_graph_embedding_model_node"
            ),
        ]
    )

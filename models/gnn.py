"""This file defines classes for the graph neural network. """
import tensorflow as tf

def graph_scatter_max_fn(point_features, point_centers, num_centers):
    aggregated = tf.math.unsorted_segment_max(point_features,
        point_centers, num_centers, name='scatter_max')
    return aggregated

def graph_scatter_sum_fn(point_features, point_centers, num_centers):
    aggregated = tf.math.unsorted_segment_sum(point_features,
        point_centers, num_centers, name='scatter_sum')
    return aggregated

def graph_scatter_mean_fn(point_features, point_centers, num_centers):
    aggregated = tf.math.unsorted_segment_mean(point_features,
        point_centers, num_centers, name='scatter_mean')
    return aggregated

def none(point_features, point_centers, num_centers):
    return point_features

### classes for implementing mini-gnn
class MLP_8_16_32_64(tf.keras.layers.Layer):
    def __init__(self, name='dense_16_32_64', dropout_rate = 0.6):
        super(MLP_8_16_32_64, self).__init__( name=name)
        self.dense0 = tf.keras.layers.Dense(8, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(l2=0.001))
        self.dense1 = tf.keras.layers.Dense(16, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(l2=0.001))
        self.dense2 = tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(l2=0.001))
        self.dense3 = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(l2=0.001))
        
    def call(self, x_, training = True):
        x_ = self.dense0(x_)
        x_ = self.dense1(x_)
        x_ = self.dense2(x_)
        x_ = self.dense3(x_)
        return x_

class MLP_64_64(tf.keras.layers.Layer):
    def __init__(self, name='dense_64_64', dropout_rate = 0.9):
        super(MLP_64_64, self).__init__( name=name)
        self.dense1 = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(l2=0.001))
        self.dense2 = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(l2=0.001))

    def call(self, x_, training = True):
        x_ = self.dense1(x_)
        x_ = self.dense2(x_)
        return x_

class MLP_128_300(tf.keras.layers.Layer):
    def __init__(self, name='dense_64_64', dropout_rate = 0.9):
        super(MLP_128_300, self).__init__( name=name)
        self.dense1 = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(l2=0.001))
        self.dense2 = tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(l2=0.001))

    def call(self, x_, training = True):
        x_ = self.dense1(x_)
        x_ = self.dense2(x_)        
        return x_

class MLP_128_64(tf.keras.layers.Layer):
    def __init__(self, name='dense_64_64', dropout_rate = 0.9):
        super(MLP_128_64, self).__init__( name=name)
        self.dense1 = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(l2=0.001))
        self.dense2 = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(l2=0.001))

    def call(self, x_, training = True):
        x_ = self.dense1(x_)
        x_ = self.dense2(x_)
        return x_

class MLP_300_300(tf.keras.layers.Layer):
    def __init__(self, name='dense_300_300', dropout_rate = 0.8):
        super(MLP_300_300, self).__init__( name=name)
        self.dense1 = tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(l2=0.001))
        self.dense2 = tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(l2=0.001))


    def call(self, x_, training = True):
        x_ = self.dense1(x_)
        x_ = self.dense2(x_)
        return x_

class MLP_64_3(tf.keras.layers.Layer):
    def __init__(self, name='dense_64_3', dropout_rate = 0.9):
        super(MLP_64_3, self).__init__( name=name)
        self.dense1 = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(l2=0.001))
        self.dense2 = tf.keras.layers.Dense(3, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(l2=0.001))

    def call(self, x_, training = True):
        x_ = self.dense1(x_)
        x_ = self.dense2(x_)
        return x_

class MLP_64_1(tf.keras.layers.Layer):
    def __init__(self, name='dense_64_3', dropout_rate = 0.9):
        super(MLP_64_1, self).__init__( name=name)
        self.dense1 = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(l2=0.001))
        self.dense2 = tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.L2(l2=0.001))
        
    def call(self, x_, training = True):
        x_ = self.dense1(x_)
        x_ = self.dense2(x_)
        return x_
        
class Classifier(tf.keras.layers.Layer):
    def __init__(self, num_classes, name = 'classifier'):
        super(Classifier, self).__init__(name=name,)
        self.dense1 = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(l2=0.001))
        self.dense2 = tf.keras.layers.Dense(num_classes, activation='softmax', kernel_regularizer=tf.keras.regularizers.L2(l2=0.001))
    def call(self, x_, training = True):
        x_ = self.dense1(x_)
        x_ = self.dense2(x_)
        return x_
##################################################################################
class Feature_extract(tf.keras.layers.Layer):
    def __init__(self, name='feature_extract_correct',
                point_feature_fn = MLP_8_16_32_64(),
                ):
        super(Feature_extract, self).__init__(name=name)
        self.point_feature_fn = point_feature_fn

    def call(self, features, points, cluster_centers, labels,
                    training=True):
        center_coords = tf.gather(cluster_centers, labels)

        point_features = tf.concat(
            [features, points, center_coords], axis = -1
        )

        extracted_features = self.point_feature_fn(
            point_features, training=training
        )
        return extracted_features

class Mini_GNN(tf.keras.layers.Layer):
    def __init__(self, name='mini_gnn',
                edge_feature_fn = MLP_64_64(),
                aggregate_fn = graph_scatter_mean_fn,
                output_fn = MLP_64_64(),
                use_prev_feature = False):
        super(Mini_GNN, self).__init__(name=name)
        self.edge_feature_fn = edge_feature_fn
        self.aggregate_fn = aggregate_fn
        self.output_fn = output_fn
        self.use_prev = use_prev_feature

    def call(self, features,
                    points,
                    l0_edges, training=True):
        point_features = tf.gather(features, l0_edges[:,0])
        dest_features = tf.gather(features, l0_edges[:,1])
        source_coords = tf.gather(points, l0_edges[:,0])
        dest_coords = tf.gather(points, l0_edges[:,1])

        edge_features = tf.concat(
            [point_features, dest_features, source_coords-dest_coords], axis = -1
        )
        extracted_features = self.edge_feature_fn(
            edge_features, training=training
        )
        aggregated_edge_features = self.aggregate_fn(
            extracted_features, l0_edges[:,1], tf.shape(points)[0]
        ) 
        if self.use_prev:
            aggregated_edge_features = tf.concat(
                [aggregated_edge_features, features], axis = -1
            )
        updated_features = self.output_fn(aggregated_edge_features, training=training)
        output_features = updated_features + features
        return output_features

class GNN(tf.keras.layers.Layer):
    def __init__(self, name = 'GNN',
                edge_feature_fn = MLP_300_300(),
                aggregate_fn = graph_scatter_mean_fn,
                output_fn = MLP_300_300(),
                use_prev_feature = False
                ):
        super(GNN, self).__init__(name=name)
        self.edge_feature_fn = edge_feature_fn
        self.aggregate_fn = aggregate_fn
        self.output_fn = output_fn
        self.use_prev = use_prev_feature
    def call(self,  features,
                    cluster_centers,
                    edges, training=True):
        point_features = tf.gather(features, edges[:,0])
        dest_features = tf.gather(features, edges[:,1])
        source_coords = tf.gather(cluster_centers, edges[:,0])
        dest_coords = tf.gather(cluster_centers, edges[:,1])

        edge_features = tf.concat(
            [point_features, dest_features, source_coords-dest_coords], axis = -1
        )
        extracted_features = self.edge_feature_fn(
            edge_features, training = training
        )
        aggregated_edge_features = self.aggregate_fn(
            extracted_features, edges[:,1], tf.shape(cluster_centers)[0]
        ) 
        if self.use_prev:
            aggregated_edge_features = tf.concat(
                [aggregated_edge_features, features], axis = -1
            )
        updated_features = self.output_fn(
            aggregated_edge_features, training=training)
        output_features = updated_features + features
        return output_features

class Mini_to_Large(tf.keras.layers.Layer):
    def __init__(self, name='mini_to_large',
                    aggregate_fn = graph_scatter_sum_fn,
                    attention_fn = MLP_64_1(),
                    output_fn = MLP_128_300(),
                    is_attention = True):
        super(Mini_to_Large, self).__init__(name=name)
        self.aggregate_fn = aggregate_fn
        self.output_fn = output_fn
        self.attention_fn = attention_fn
        self.is_attention = is_attention
    def call(self, point_features,
                    labels,
                    cluster_centers,
                    points,
                    training= True
                    ):
        point_center_coords = tf.gather(cluster_centers, labels)
        t_features = tf.concat(
            [point_features, point_center_coords - points], axis = -1
        )
        if self.is_attention:
            t_attentions = self.attention_fn(t_features)
            t_features = tf.math.multiply(t_features, t_attentions)
        aggregated_features = self.aggregate_fn(
            t_features, labels, tf.shape(cluster_centers)[0]
        )
        output_features = self.output_fn(aggregated_features, training = training)
        return output_features

class Large_to_Mini(tf.keras.layers.Layer):
    def __init__(self, name = 'large_to_mini',
                output_feature_fn = MLP_128_64()):
        super(Large_to_Mini, self).__init__(name=name)
        self.output_feature_fn = output_feature_fn
    def call(self, features, labels, cluster_centers, points, training=True):
        center_features = tf.gather(features, labels)
        point_center_coords = tf.gather(cluster_centers, labels)
        temp_point_features = tf.concat(
            [center_features, point_center_coords - points], axis = -1
        )
        output_features = self.output_feature_fn(temp_point_features, training = training)    
        return output_features
###################################################################################
### Trying with l1 only
class FFN(tf.keras.layers.Layer):
    def __init__(self, name='mini_to_large',
                    aggregate_fn = graph_scatter_sum_fn,
                    edge_fn = MLP_8_16_32_64(),
                    attention_fn = MLP_64_1(),
                    output_fn = MLP_128_300(),
                    is_attention = True):
        super(FFN, self).__init__(name=name)
        self.aggregate_fn = aggregate_fn
        self.output_fn = output_fn
        self.attention_fn = attention_fn
        self.edge_fn = edge_fn
        self.is_attention = is_attention
    def call(self, point_features,
                    labels,
                    cluster_centers,
                    points,
                    training= True
                    ):
        point_center_coords = tf.gather(cluster_centers, labels)
        temp_point_features = tf.concat(
            [point_features, point_center_coords - points], axis = -1
        )
        t_features = self.edge_fn(temp_point_features)
        if self.is_attention:
            t_attentions = self.attention_fn(temp_point_features)
            t_features = tf.math.multiply(t_features, t_attentions)
        aggregated_features = self.aggregate_fn(
            t_features, labels, tf.shape(cluster_centers)[0]
        )
        output_features = self.output_fn(aggregated_features, training = training)
        return output_features

class FBN(tf.keras.layers.Layer):
    def __init__(self, name = 'large_to_mini',
                output_feature_fn = MLP_128_64()):
        super(FBN, self).__init__(name=name)
        self.output_feature_fn = output_feature_fn
    def call(self, features, labels, cluster_centers, points, training=True):
        center_features = tf.gather(features, labels)
        point_center_coords = tf.gather(cluster_centers, labels)
        temp_point_features = tf.concat(
            [center_features, point_center_coords - points], axis = -1
        )
        output_features = self.output_feature_fn(temp_point_features, training = training)    
        return output_features

class GNN_(tf.keras.layers.Layer):
    def __init__(self, name = 'GNN',
                edge_feature_fn = MLP_300_300(),
                aggregate_fn = graph_scatter_mean_fn,
                output_fn = MLP_300_300()
                ):
        super(GNN_, self).__init__(name=name)
        self.edge_feature_fn = edge_feature_fn
        self.aggregate_fn = aggregate_fn
        self.output_fn = output_fn
       
    def call(self,  features,
                    cluster_centers,
                    l1_edges, training=True):
        point_features = tf.gather(features, l1_edges[:,0])
        dest_features = tf.gather(features, l1_edges[:,1])
        source_coords = tf.gather(cluster_centers, l1_edges[:,0])
        dest_coords = tf.gather(cluster_centers, l1_edges[:,1])

        edge_features = tf.concat(
            [point_features, dest_features, source_coords-dest_coords], axis = -1
        )
        extracted_features = self.edge_feature_fn(
            edge_features, training = training
        )
        aggregated_edge_features = self.aggregate_fn(
            extracted_features, l1_edges[:,1], tf.shape(cluster_centers)[0]
        ) 
        updated_features = self.output_fn(aggregated_edge_features, training=training)
        output_features = updated_features + features
        return output_features

###################################################
###################################################
'''Making reconfigurable model layers'''
###################################################
###################################################

def create_mlp_with_BN(hidden_units, activation = tf.nn.gelu ,dropout_rate = 0.8, name='test'):
    '''A function to provide sequential dense layers with dropout and batchNorm'''
    mlp_layers = []
    for units in hidden_units:
        mlp_layers.append(tf.keras.layers.BatchNormalization())
        mlp_layers.append(tf.keras.layers.Dropout(dropout_rate))
        mlp_layers.append(tf.keras.layers.Dense(units, activation=activation))
    return tf.keras.Sequential(mlp_layers, name=name)

def create_mlp(hidden_units, activation = tf.nn.gelu ,dropout_rate = 0.8, name='test'):
    '''A function to provide sequential dense layers with dropout and batchNorm'''
    mlp_layers = []
    for units in hidden_units:
        mlp_layers.append(tf.keras.layers.Dense(units, activation=activation))
        mlp_layers.append(tf.keras.layers.Dropout(dropout_rate))
    return tf.keras.Sequential(mlp_layers, name=name)

class FeatureFeedForward(tf.keras.layers.Layer):
    def __init__(self, name='feature_extract',
                edge_feature_mlp = [16,32,64],
                aggregate_fn = 'mean',
                output_mlp = [64,64],
                ):
        super(FeatureFeedForward, self).__init__(name=name)
        self.edge_feature_fn = create_mlp(edge_feature_mlp, name='ffn_edge')
        self.aggregate_fn = get_agg_func(aggregate_fn)
        self.output_fn = create_mlp(output_mlp, name='ffn_out')

    def call(self, features,
                    points,
                    l0_edges, training = True):
        point_features = tf.gather(features, l0_edges[:,0])
        dest_features = tf.gather(features, l0_edges[:,1])
        source_coords = tf.gather(points, l0_edges[:,0])
        dest_coords = tf.gather(points, l0_edges[:,1])

        edge_features = tf.concat(
            [point_features, dest_features, (source_coords-dest_coords)], axis = -1
        )
        extracted_features = self.edge_feature_fn(
            edge_features, training = training
        )
        aggregated_edge_features = self.aggregate_fn(
            extracted_features, l0_edges[:,1], tf.shape(points)[0]
        )
        updated_features = self.output_fn(aggregated_edge_features, training = training) 
        
        return updated_features


class LocalGNN(tf.keras.layers.Layer):
    def __init__(self, name='local_gnn',
                edge_feature_mlp = [64,64],
                aggregate_fn = 'mean',
                auto_offset_mlp = [64,3],
                output_mlp = [64,64],
                auto_offset= True):
        super(LocalGNN, self).__init__(name=name)
        self.edge_feature_fn = create_mlp(edge_feature_mlp, name='edge')
        self.auto_offset_fn = create_mlp(auto_offset_mlp)
        self.auto_offset = auto_offset
        self.aggregate_fn = get_agg_func(aggregate_fn)
        self.output_fn = create_mlp(output_mlp)

    def call(self, features,
                    points,
                    l0_edges, training = True):
        point_features = tf.gather(features, l0_edges[:,0])
        dest_features = tf.gather(features, l0_edges[:,1])
        source_coords = tf.gather(points, l0_edges[:,0])
        dest_coords = tf.gather(points, l0_edges[:,1])

        if self.auto_offset:
            offset = self.auto_offset_fn(point_features, training = training)
            source_coords = source_coords + offset
        edge_features = tf.concat(
            [point_features, dest_features, source_coords-dest_coords], axis = -1
        )
        extracted_features = self.edge_feature_fn(
            edge_features, training=training
        )
        aggregated_edge_features = self.aggregate_fn(
            extracted_features, l0_edges[:,1], tf.shape(points)[0]
        )
        updated_features = self.output_fn(aggregated_edge_features, training=training)
        output_features = updated_features + features
        return output_features


class MiniToLarge(tf.keras.layers.Layer):
    def __init__(self, name='mini_to_large',
                    aggregate_fn = 'mean',
                    output_mlp = [300,300]):
        super(MiniToLarge, self).__init__(name=name)
        self.aggregate_fn = get_agg_func(aggregate_fn)
        self.output_fn = create_mlp(output_mlp)
    
    def call(self, point_features,
                    labels,
                    cluster_centers,
                    training
                    ):
        aggregated_features = self.aggregate_fn(
            point_features, labels, tf.shape(cluster_centers)[0]
        )
        output_features = self.output_fn(aggregated_features, training = training)
        return output_features

class GlobalGNN(tf.keras.layers.Layer):
    def __init__(self, name = 'global_GNN',
                edge_feature_mlp = [300,300],
                aggregate_fn = 'mean',
                auto_offset_mlp = [64,3],
                output_mlp = [300,300],
                auto_offset= True):
        super(GlobalGNN, self).__init__(name=name)
        self.edge_feature_fn = create_mlp(edge_feature_mlp)
        self.auto_offset_fn = create_mlp(auto_offset_mlp)
        self.auto_offset = auto_offset
        self.aggregate_fn = get_agg_func(aggregate_fn)
        self.output_fn = create_mlp(output_mlp)
       
    def call(self,  features,
                    cluster_centers,
                    l1_edges,
                    training = True):
        point_features = tf.gather(features, l1_edges[:,0])
        dest_features = tf.gather(features, l1_edges[:,1])
        source_coords = tf.gather(cluster_centers, l1_edges[:,0])
        dest_coords = tf.gather(cluster_centers, l1_edges[:,1])

        if self.auto_offset:
            offset = self.auto_offset_fn(point_features, training=training)
            source_coords = source_coords + offset
        edge_features = tf.concat(
            [point_features, dest_features, source_coords-dest_coords], axis = -1
        )
        extracted_features = self.edge_feature_fn(
            edge_features, training = training
        )
        aggregated_edge_features = self.aggregate_fn(
            extracted_features, l1_edges[:,1], tf.shape(cluster_centers)[0]
        )
        updated_features = self.output_fn(aggregated_edge_features, training=training)
        output_features = updated_features + features
        return output_features

class LargeToMini(tf.keras.layers.Layer):
    def __init__(self, name = 'large_to_mini',
                output_feature_mlp = [64,64]):
        super(LargeToMini, self).__init__(name=name)
        self.output_feature_fn = create_mlp(output_feature_mlp)

    def call(self, features, labels, training = True):
        output_features = tf.gather(features, labels)
        output_features = self.output_feature_fn(output_features, training = training)
        return output_features


def get_agg_func(func):
    functions = {
        'sum': graph_scatter_sum_fn,
        'mean': graph_scatter_mean_fn,
        'max': graph_scatter_max_fn
    }
    return functions[func]

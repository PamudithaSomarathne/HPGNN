#from typing import final
import tensorflow as tf

import models.gnn as gnn  

class Mini_pointgnn(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super(Mini_pointgnn, self).__init__(*args, **kwargs)
        self.layer1 = gnn.Feature_extract() #5
        self.layer2 = gnn.Mini_GNN() #6
        self.layer3 = gnn.Mini_to_Large() #2
        self.layer4 = gnn.GNN() #6
        self.layer5 = gnn.Large_to_Mini() #2
        self.layer6 = gnn.Mini_GNN() #6

        self.classifier = gnn.Classifier()

    def call(self, inputs, training=False):
        features, points, cluster_centers, l0_edges, l1_edges, labels = inputs
        t_features_1 = self.layer1(
            features, points, cluster_centers, labels, training
        )
        
        t_features_2 = self.layer2(
            t_features_1, points, l0_edges, training
        )
        
        t_features_3 = self.layer3(
            t_features_2, labels, cluster_centers, training
        )
        
        t_features_4 = self.layer4(
            t_features_3, cluster_centers, l1_edges, training
        )
        
        t_features_5 = self.layer5(
            t_features_4, labels, training
        )
        
        t_features_6 = self.layer6(
            t_features_5, points, l0_edges, training
        )
        final_features = t_features_6 + t_features_2
        final_features = self.classifier(final_features)
        
        return final_features

class MultiScaleGNN(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super(MultiScaleGNN, self).__init__(*args, **kwargs)
        self.layer1 = gnn.FeatureFeedForward(name='feature_extract',
                                            edge_feature_mlp = [16,32,64],
                                            aggregate_fn = 'mean',
                                            output_mlp = [64,64]) 
        self.layer2 = gnn.LocalGNN(name='local_gnn_0',
                                    edge_feature_mlp = [64,64],
                                    aggregate_fn = 'mean',
                                    auto_offset_mlp = [64,3],
                                    output_mlp = [64,64],
                                    auto_offset= False) 
        self.layer3 = gnn.MiniToLarge(name='mini_to_large',
                                        aggregate_fn = 'mean',
                                        output_mlp = [128,256]) 
        self.layer4 = gnn.GlobalGNN(name = 'global_GNN_0',
                                        edge_feature_mlp = [256,256],
                                        aggregate_fn = 'max',
                                        auto_offset_mlp = [64,3],
                                        output_mlp = [256,256],
                                        auto_offset= True)
        self.layer4_1 = gnn.GlobalGNN(name = 'global_GNN_1',
                                        edge_feature_mlp = [256,256],
                                        aggregate_fn = 'max',
                                        auto_offset_mlp = [64,3],
                                        output_mlp = [256,256],
                                        auto_offset= True) 
        self.layer5 = gnn.LargeToMini(name = 'large_to_mini',
                                        output_feature_mlp = [128,64]) 
        self.layer6 = gnn.LocalGNN(name='local_gnn_1',
                                    edge_feature_mlp = [64,64],
                                    aggregate_fn = 'mean',
                                    auto_offset_mlp = [64,3],
                                    output_mlp = [64,64],
                                    auto_offset= False) 
        # self.classifier = tf.keras.Sequential([
        #     tf.keras.layers.Dense(64, activation = 'relu'),
        #     tf.keras.layers.Dense(20, activation = 'softmax')
        # ], name='classifier')
        self.classifier = gnn.Classifier()
    
    def call(self, inputs, training = True):
        features, points, cluster_centers, l0_edges, l1_edges, labels = inputs
        t_features_1 = self.layer1(
            features, points, l0_edges, training
        )
        
        t_features_2 = self.layer2(
            t_features_1, points, l0_edges, training
        )
        
        t_features_3 = self.layer3(
            t_features_2, labels, cluster_centers, training
        )
        
        t_features_4 = self.layer4(
            t_features_3, cluster_centers, l1_edges, training
        )
        t_features_4 = self.layer4_1(
            t_features_4, cluster_centers, l1_edges, training
        )
        t_features_5 = self.layer5(
            t_features_4, labels, training
        )
        
        t_features_6 = self.layer6(
            t_features_5, points, l0_edges, training
        )
        final_features = t_features_6 + t_features_2
        final_features = self.classifier(final_features)
        
        return final_features
        

class Mini_pointgnn_v1(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super(Mini_pointgnn_v1, self).__init__(*args, **kwargs)
        self.layer1 = gnn.Feature_extract() #5
        self.layer2 = gnn.Mini_GNN(aggregate_fn=gnn.graph_scatter_max_fn) #6
        self.layer3 = gnn.Mini_to_Large(aggregate_fn=gnn.graph_scatter_max_fn) #2
        self.layer4 = gnn.GNN(aggregate_fn=gnn.graph_scatter_max_fn) #6
        self.layer5 = gnn.Large_to_Mini() #2
        self.layer6 = gnn.Mini_GNN(aggregate_fn=gnn.graph_scatter_max_fn) #6

        self.classifier = gnn.Classifier()

    def call(self, inputs, training=False):
        features, points, cluster_centers, l0_edges, l1_edges, labels = inputs
        t_features_1 = self.layer1(
            features, points, cluster_centers, labels, training
        )
        
        t_features_2 = self.layer2(
            t_features_1, points, l0_edges, training
        )
        
        t_features_3 = self.layer3(
            t_features_2, labels, cluster_centers, points, training
        )
        
        t_features_4 = self.layer4(
            t_features_3, cluster_centers, l1_edges, training
        )
        
        t_features_5 = self.layer5(
            t_features_4, labels, cluster_centers, points, training
        )
        
        t_features_6 = self.layer6(
            t_features_5, points, l0_edges, training
        )
        final_features = t_features_6 + t_features_2
        final_features = self.classifier(final_features)
        
        return final_features


class Mini_pointgnn_v2(tf.keras.Model):
    '''same model as 2.0 but uses max funxtion'''
    def __init__(self, *args, **kwargs):
        super(Mini_pointgnn_v2, self).__init__(*args, **kwargs)
        self.layer1 = gnn.Feature_extract() #5
        self.layer2 = gnn.Mini_GNN(aggregate_fn=gnn.graph_scatter_max_fn, use_prev_feature=True) #6
        self.layer2_1 = gnn.Mini_GNN(aggregate_fn=gnn.graph_scatter_max_fn, use_prev_feature=True) #6
        self.layer3 = gnn.Mini_to_Large(aggregate_fn=gnn.graph_scatter_sum_fn) #2
        self.layer4 = gnn.GNN(aggregate_fn=gnn.graph_scatter_max_fn, use_prev_feature=True) #6
        self.layer4_1 = gnn.GNN(aggregate_fn=gnn.graph_scatter_max_fn, use_prev_feature=True) #6
        self.layer5 = gnn.Large_to_Mini() #2
        self.layer6 = gnn.Mini_GNN(aggregate_fn=gnn.graph_scatter_max_fn, use_prev_feature=True) #6
        self.layer6_1 = gnn.Mini_GNN(aggregate_fn=gnn.graph_scatter_max_fn, use_prev_feature=True) #6
        self.classifier = gnn.Classifier()

    def call(self, inputs, training=False):
        features, points, cluster_centers, l0_edges, l1_edges, labels = inputs
        t_features_1 = self.layer1(
            features, points, cluster_centers, labels, training
        )
        
        t_features_2 = self.layer2(
            t_features_1, points, l0_edges, training
        )
        t_features_2_1 = self.layer2_1(
            t_features_2, points, l0_edges, training
        )
        
        t_features_3 = self.layer3(
            t_features_2_1, labels, cluster_centers, points, training
        )
        
        t_features_4 = self.layer4(
            t_features_3, cluster_centers, l1_edges, training
        )
        t_features_4_1 = self.layer4_1(
            t_features_4, cluster_centers, l1_edges, training
        )
        
        t_features_5 = self.layer5(
            t_features_4_1, labels, cluster_centers, points, training
        )
        
        t_features_6 = self.layer6(
            t_features_5, points, l0_edges, training
        )
        t_features_6 = t_features_6 + t_features_2_1
        t_features_6_1 = self.layer6_1(
            t_features_6, points, l0_edges, training
        )
        final_features = t_features_6_1 + t_features_2
        final_features = self.classifier(final_features)
        
        return final_features


class Mini_pointgnn_v3(tf.keras.Model):
    '''121'''
    def __init__(self, *args, **kwargs):
        super(Mini_pointgnn_v3, self).__init__(*args, **kwargs)
        self.layer1 = gnn.Feature_extract() #5
        self.layer2 = gnn.Mini_GNN(aggregate_fn=gnn.graph_scatter_max_fn) #6
        self.layer3 = gnn.Mini_to_Large(aggregate_fn=gnn.graph_scatter_sum_fn) #2
        self.layer4 = gnn.GNN(aggregate_fn=gnn.graph_scatter_max_fn) #6
        self.layer4_1 = gnn.GNN(aggregate_fn=gnn.graph_scatter_max_fn) #6
        self.layer5 = gnn.Large_to_Mini() #2
        self.layer6 = gnn.Mini_GNN(aggregate_fn=gnn.graph_scatter_max_fn) #6
        self.classifier = gnn.Classifier()

    def call(self, inputs, training=False):
        features, points, cluster_centers, l0_edges, l1_edges, labels = inputs
        t_features_1 = self.layer1(
            features, points, cluster_centers, labels, training
        )
        
        t_features_2 = self.layer2(
            t_features_1, points, l0_edges, training
        )
        
        t_features_3 = self.layer3(
            t_features_2, labels, cluster_centers, points, training
        )
        
        t_features_4 = self.layer4(
            t_features_3, cluster_centers, l1_edges, training
        )

        t_features_4 = self.layer4_1(
            t_features_4, cluster_centers, l1_edges, training
        )

        t_features_5 = self.layer5(
            t_features_4, labels, cluster_centers, points, training
        )
        
        t_features_6 = self.layer6(
            t_features_5, points, l0_edges, training
        )
        final_features = t_features_6 + t_features_2
        final_features = self.classifier(final_features)
        
        return final_features

class Mini_pointgnn_v4(tf.keras.Model):
    '''122'''
    def __init__(self, *args, **kwargs):
        super(Mini_pointgnn_v4, self).__init__(*args, **kwargs)
        self.layer1 = gnn.Feature_extract() #5
        self.layer2 = gnn.Mini_GNN(aggregate_fn=gnn.graph_scatter_max_fn) #6
        self.layer3 = gnn.Mini_to_Large(aggregate_fn=gnn.graph_scatter_sum_fn) #2
        self.layer4 = gnn.GNN(aggregate_fn=gnn.graph_scatter_max_fn) #6
        self.layer4_1 = gnn.GNN(aggregate_fn=gnn.graph_scatter_max_fn) #6
        self.layer5 = gnn.Large_to_Mini() #2
        self.layer6 = gnn.Mini_GNN(aggregate_fn=gnn.graph_scatter_max_fn) #6
        self.layer6_1 = gnn.Mini_GNN(aggregate_fn=gnn.graph_scatter_max_fn) #6
        self.classifier = gnn.Classifier()

    def call(self, inputs, training=False):
        features, points, cluster_centers, l0_edges, l1_edges, labels = inputs
        t_features_1 = self.layer1(
            features, points, cluster_centers, labels, training
        )
        
        t_features_2 = self.layer2(
            t_features_1, points, l0_edges, training
        )
        
        t_features_3 = self.layer3(
            t_features_2, labels, cluster_centers, points, training
        )
        
        t_features_4 = self.layer4(
            t_features_3, cluster_centers, l1_edges, training
        )

        t_features_4 = self.layer4_1(
            t_features_4, cluster_centers, l1_edges, training
        )

        t_features_5 = self.layer5(
            t_features_4, labels, cluster_centers, points, training
        )
        
        t_features_6 = self.layer6(
            t_features_5, points, l0_edges, training
        )
        t_features_6 = self.layer6_1(
            t_features_6, points, l0_edges, training
        )
        final_features = t_features_6 + t_features_2
        final_features = self.classifier(final_features)
        
        return final_features

class Mini_pointgnn_v5(tf.keras.Model):
    '''132'''
    def __init__(self, *args, **kwargs):
        super(Mini_pointgnn_v5, self).__init__(*args, **kwargs)
        self.layer1 = gnn.Feature_extract() #5
        self.layer2 = gnn.Mini_GNN(aggregate_fn=gnn.graph_scatter_max_fn) #6
        self.layer3 = gnn.Mini_to_Large(aggregate_fn=gnn.graph_scatter_sum_fn) #2
        self.layer4 = gnn.GNN(aggregate_fn=gnn.graph_scatter_max_fn) #6
        self.layer4_1 = gnn.GNN(aggregate_fn=gnn.graph_scatter_max_fn) #6
        self.layer4_2 = gnn.GNN(aggregate_fn=gnn.graph_scatter_max_fn) #6
        self.layer5 = gnn.Large_to_Mini() #2
        self.layer6 = gnn.Mini_GNN(aggregate_fn=gnn.graph_scatter_max_fn) #6
        self.layer6_1 = gnn.Mini_GNN(aggregate_fn=gnn.graph_scatter_max_fn) #6
        self.classifier = gnn.Classifier()

    def call(self, inputs, training=False):
        features, points, cluster_centers, l0_edges, l1_edges, labels = inputs
        t_features_1 = self.layer1(
            features, points, cluster_centers, labels, training
        )
        
        t_features_2 = self.layer2(
            t_features_1, points, l0_edges, training
        )
        
        t_features_3 = self.layer3(
            t_features_2, labels, cluster_centers, points, training
        )
        
        t_features_4 = self.layer4(
            t_features_3, cluster_centers, l1_edges, training
        )

        t_features_4 = self.layer4_1(
            t_features_4, cluster_centers, l1_edges, training
        )
        t_features_4 = self.layer4_2(
            t_features_4, cluster_centers, l1_edges, training
        )
        t_features_5 = self.layer5(
            t_features_4, labels, cluster_centers, points, training
        )
        
        t_features_6 = self.layer6(
            t_features_5, points, l0_edges, training
        )
        t_features_6 = self.layer6_1(
            t_features_6, points, l0_edges, training
        )
        final_features = t_features_6 + t_features_2
        final_features = self.classifier(final_features)
        
        return final_features

class Mini_pointgnn_v6(tf.keras.Model):
    '''
    No l0edges
    '''
    def __init__(self, *args, **kwargs):
        super(Mini_pointgnn_v6, self).__init__(*args, **kwargs)
        self.layer1 = gnn.FFN(aggregate_fn=gnn.graph_scatter_sum_fn) 
        self.layer2 = gnn.GNN(aggregate_fn=gnn.graph_scatter_max_fn) 
        self.layer2_1 = gnn.GNN(aggregate_fn=gnn.graph_scatter_max_fn)
        self.layer3 = gnn.FBN() 
        self.classifier = gnn.Classifier()

    def call(self, inputs, training=False):
        features, points, cluster_centers, l0_edges, l1_edges, labels = inputs
        t_features_1 = self.layer1(
            features, labels, cluster_centers, points
        )
        
        t_features_2 = self.layer2(
            t_features_1, cluster_centers, l1_edges, training
        )
        t_features_2_1 = self.layer2_1(
            t_features_2, cluster_centers, l1_edges, training
        )
        t_features_3 = self.layer3(
            t_features_2_1, labels, cluster_centers, points, training
        )
        final_features = self.classifier(t_features_3)
        return final_features

class Mini_pointgnn_v7(tf.keras.Model):
    '''
    No l0edges, 3 levels
    1-2-1
    '''
    def __init__(self, num_classes, *args, **kwargs):
        super(Mini_pointgnn_v7, self).__init__(*args, **kwargs)
        self.layer1 = gnn.FFN(aggregate_fn=gnn.graph_scatter_sum_fn,
                                output_fn=gnn.MLP_64_64(),
                                is_attention=False) 
        self.layer2 = gnn.GNN(aggregate_fn=gnn.graph_scatter_max_fn,
                                edge_feature_fn=gnn.MLP_64_64(),
                                output_fn=gnn.MLP_64_64()) 
        self.layer3 = gnn.Mini_to_Large(is_attention=False)
        self.layer4 = gnn.GNN(aggregate_fn=gnn.graph_scatter_max_fn)
        self.layer4_1 = gnn.GNN(aggregate_fn=gnn.graph_scatter_max_fn)
        self.layer5 = gnn.Large_to_Mini()
        self.layer6 = gnn.GNN(aggregate_fn=gnn.graph_scatter_max_fn,
                                edge_feature_fn=gnn.MLP_64_64(),
                                output_fn=gnn.MLP_64_64()) 
        self.layer7 = gnn.FBN() 
        self.classifier = gnn.Classifier(num_classes)

    def call(self, inputs, training=False):
        remission, points, l1_cluster_centers, l2_cluster_centers, \
           l1_edges, l2_edges, l1_labels, l2_labels = inputs
        t_features_1 = self.layer1(
            remission, l1_labels, l1_cluster_centers, points
        )

        t_features_2 = self.layer2(
            t_features_1, l1_cluster_centers, l1_edges, training
        )
        t_features_3 = self.layer3(
            t_features_2, l2_labels, l2_cluster_centers, l1_cluster_centers, training
        )
        t_features_4 = self.layer4(
            t_features_3, l2_cluster_centers, l2_edges, training
        )
        t_features_4 = self.layer4_1(
            t_features_4, l2_cluster_centers, l2_edges, training
        )
        t_features_5 = self.layer5(
            t_features_4, l2_labels, l2_cluster_centers, l1_cluster_centers, training
        )
        t_features_6 = self.layer6(
            t_features_5, l1_cluster_centers, l1_edges, training
        )
        t_features_6 = t_features_6 + t_features_2
        t_features_7 = self.layer7(
            t_features_6, l1_labels, l1_cluster_centers, points, training
        )
        final_features = self.classifier(t_features_7)
        return final_features

class Mini_pointgnn_v8(tf.keras.Model):
    '''
    1-0-1
    '''
    def __init__(self, *args, **kwargs):
        super(Mini_pointgnn_v8, self).__init__(*args, **kwargs)
        self.layer1 = gnn.FFN(aggregate_fn=gnn.graph_scatter_sum_fn,
                                output_fn=gnn.MLP_64_64()) 
        self.layer2 = gnn.GNN(aggregate_fn=gnn.graph_scatter_max_fn,
                                edge_feature_fn=gnn.MLP_64_64(),
                                output_fn=gnn.MLP_64_64()) 
        # self.layer3 = gnn.Mini_to_Large()
        # self.layer4 = gnn.GNN(aggregate_fn=gnn.graph_scatter_max_fn)
        # self.layer4_1 = gnn.GNN(aggregate_fn=gnn.graph_scatter_max_fn)
        # self.layer5 = gnn.Large_to_Mini()
        self.layer6 = gnn.GNN(aggregate_fn=gnn.graph_scatter_max_fn,
                                edge_feature_fn=gnn.MLP_64_64(),
                                output_fn=gnn.MLP_64_64()) 
        self.layer7 = gnn.FBN() 
        self.classifier = gnn.Classifier()

    def call(self, inputs, training=False):
        remission, points, l1_cluster_centers, l2_cluster_centers, \
           l1_edges, l2_edges, l1_labels, l2_labels = inputs
        t_features_1 = self.layer1(
            remission, l1_labels, l1_cluster_centers, points
        )
        
        t_features_2 = self.layer2(
            t_features_1, l1_cluster_centers, l1_edges, training
        )
        # t_features_2 = self.layer2_1(
        #     t_features_2, l1_cluster_centers, l1_edges, training
        # )
        # t_features_3 = self.layer3(
        #     t_features_2, l2_labels, l2_cluster_centers, l1_cluster_centers, training
        # )
        # t_features_4 = self.layer4(
        #     t_features_3, l2_cluster_centers, l2_edges, training
        # )
        # t_features_4 = self.layer4_1(
        #     t_features_4, l2_cluster_centers, l2_edges, training
        # )
        # t_features_5 = self.layer5(
        #     t_features_4, l2_labels, l2_cluster_centers, l1_cluster_centers, training
        # )
        t_features_6 = self.layer6(
            t_features_2, l1_cluster_centers, l1_edges, training
        )
        t_features_6 = t_features_6 + t_features_2
        t_features_7 = self.layer7(
            t_features_6, l1_labels, l1_cluster_centers, points, training
        )
        final_features = self.classifier(t_features_7)
        return final_features

class Mini_pointgnn_v9(tf.keras.Model):
    '''
    No l0edges, 3 levels
    1-1-1
    '''
    def __init__(self, *args, **kwargs):
        super(Mini_pointgnn_v9, self).__init__(*args, **kwargs)
        self.layer1 = gnn.FFN(aggregate_fn=gnn.graph_scatter_sum_fn,
                                output_fn=gnn.MLP_64_64()) 
        self.layer2 = gnn.GNN(aggregate_fn=gnn.graph_scatter_max_fn,
                                edge_feature_fn=gnn.MLP_64_64(),
                                output_fn=gnn.MLP_64_64()) 
        self.layer3 = gnn.Mini_to_Large()
        self.layer4 = gnn.GNN(aggregate_fn=gnn.graph_scatter_max_fn)
        # self.layer4_1 = gnn.GNN(aggregate_fn=gnn.graph_scatter_max_fn)
        self.layer5 = gnn.Large_to_Mini()
        self.layer6 = gnn.GNN(aggregate_fn=gnn.graph_scatter_max_fn,
                                edge_feature_fn=gnn.MLP_64_64(),
                                output_fn=gnn.MLP_64_64()) 
        self.layer7 = gnn.FBN() 
        self.classifier = gnn.Classifier()

    def call(self, inputs, training=False):
        remission, points, l1_cluster_centers, l2_cluster_centers, \
           l1_edges, l2_edges, l1_labels, l2_labels = inputs
        t_features_1 = self.layer1(
            remission, l1_labels, l1_cluster_centers, points
        )
        
        t_features_2 = self.layer2(
            t_features_1, l1_cluster_centers, l1_edges, training
        )
        t_features_3 = self.layer3(
            t_features_2, l2_labels, l2_cluster_centers, l1_cluster_centers, training
        )
        t_features_4 = self.layer4(
            t_features_3, l2_cluster_centers, l2_edges, training
        )
        # t_features_4 = self.layer4_1(
        #     t_features_4, l2_cluster_centers, l2_edges, training
        # )
        t_features_5 = self.layer5(
            t_features_4, l2_labels, l2_cluster_centers, l1_cluster_centers, training
        )
        t_features_6 = self.layer6(
            t_features_5, l1_cluster_centers, l1_edges, training
        )
        t_features_6 = t_features_6 + t_features_2
        t_features_7 = self.layer7(
            t_features_6, l1_labels, l1_cluster_centers, points, training
        )
        final_features = self.classifier(t_features_7)
        return final_features

class Mini_pointgnn_v10(tf.keras.Model):
    '''
    No l0edges, 3 levels
    1-3-1
    '''
    def __init__(self, *args, **kwargs):
        super(Mini_pointgnn_v10, self).__init__(*args, **kwargs)
        self.layer1 = gnn.FFN(aggregate_fn=gnn.graph_scatter_sum_fn,
                                output_fn=gnn.MLP_64_64()) 
        self.layer2 = gnn.GNN(aggregate_fn=gnn.graph_scatter_max_fn,
                                edge_feature_fn=gnn.MLP_64_64(),
                                output_fn=gnn.MLP_64_64()) 
        self.layer3 = gnn.Mini_to_Large()
        self.layer4 = gnn.GNN(aggregate_fn=gnn.graph_scatter_max_fn)
        self.layer4_1 = gnn.GNN(aggregate_fn=gnn.graph_scatter_max_fn)
        self.layer4_2 = gnn.GNN(aggregate_fn=gnn.graph_scatter_max_fn)
        self.layer5 = gnn.Large_to_Mini()
        self.layer6 = gnn.GNN(aggregate_fn=gnn.graph_scatter_max_fn,
                                edge_feature_fn=gnn.MLP_64_64(),
                                output_fn=gnn.MLP_64_64()) 
        self.layer7 = gnn.FBN() 
        self.classifier = gnn.Classifier()

    def call(self, inputs, training=False):
        remission, points, l1_cluster_centers, l2_cluster_centers, \
           l1_edges, l2_edges, l1_labels, l2_labels = inputs
        t_features_1 = self.layer1(
            remission, l1_labels, l1_cluster_centers, points
        )
        
        t_features_2 = self.layer2(
            t_features_1, l1_cluster_centers, l1_edges, training
        )
        t_features_3 = self.layer3(
            t_features_2, l2_labels, l2_cluster_centers, l1_cluster_centers, training
        )
        t_features_4 = self.layer4(
            t_features_3, l2_cluster_centers, l2_edges, training
        )
        t_features_4 = self.layer4_1(
            t_features_4, l2_cluster_centers, l2_edges, training
        )
        t_features_4 = self.layer4_2(
            t_features_4, l2_cluster_centers, l2_edges, training
        )
        t_features_5 = self.layer5(
            t_features_4, l2_labels, l2_cluster_centers, l1_cluster_centers, training
        )
        t_features_6 = self.layer6(
            t_features_5, l1_cluster_centers, l1_edges, training
        )
        t_features_6 = t_features_6 + t_features_2
        t_features_7 = self.layer7(
            t_features_6, l1_labels, l1_cluster_centers, points, training
        )
        final_features = self.classifier(t_features_7)
        return final_features

class Mini_pointgnn_v11(tf.keras.Model):
    '''
    No l0edges, 3 levels
    2-1-2
    '''
    def __init__(self, *args, **kwargs):
        super(Mini_pointgnn_v11, self).__init__(*args, **kwargs)
        self.layer1 = gnn.FFN(aggregate_fn=gnn.graph_scatter_sum_fn,
                                output_fn=gnn.MLP_64_64()) 
        self.layer2 = gnn.GNN(aggregate_fn=gnn.graph_scatter_max_fn,
                                edge_feature_fn=gnn.MLP_64_64(),
                                output_fn=gnn.MLP_64_64())
        self.layer2_1 = gnn.GNN(aggregate_fn=gnn.graph_scatter_max_fn,
                                edge_feature_fn=gnn.MLP_64_64(),
                                output_fn=gnn.MLP_64_64()) 
        self.layer3 = gnn.Mini_to_Large()
        self.layer4 = gnn.GNN(aggregate_fn=gnn.graph_scatter_max_fn)
        # self.layer4_1 = gnn.GNN(aggregate_fn=gnn.graph_scatter_max_fn)
        self.layer5 = gnn.Large_to_Mini()
        self.layer6 = gnn.GNN(aggregate_fn=gnn.graph_scatter_max_fn,
                                edge_feature_fn=gnn.MLP_64_64(),
                                output_fn=gnn.MLP_64_64()) 
        self.layer6_1 = gnn.GNN(aggregate_fn=gnn.graph_scatter_max_fn,
                                edge_feature_fn=gnn.MLP_64_64(),
                                output_fn=gnn.MLP_64_64())
        self.layer7 = gnn.FBN() 
        self.classifier = gnn.Classifier()

    def call(self, inputs, training=False):
        remission, points, l1_cluster_centers, l2_cluster_centers, \
           l1_edges, l2_edges, l1_labels, l2_labels = inputs
        t_features_1 = self.layer1(
            remission, l1_labels, l1_cluster_centers, points
        )
        
        t_features_2 = self.layer2(
            t_features_1, l1_cluster_centers, l1_edges, training
        )
        t_features_2_1 = self.layer2_1(
            t_features_2, l1_cluster_centers, l1_edges, training
        )
        t_features_3 = self.layer3(
            t_features_2_1, l2_labels, l2_cluster_centers, l1_cluster_centers, training
        )
        t_features_4 = self.layer4(
            t_features_3, l2_cluster_centers, l2_edges, training
        )
        t_features_5 = self.layer5(
            t_features_4, l2_labels, l2_cluster_centers, l1_cluster_centers, training
        )
        t_features_6 = self.layer6(
            t_features_5, l1_cluster_centers, l1_edges, training
        )
        t_features_6 = t_features_6 + t_features_2_1
        t_features_6 = self.layer6_1(
            t_features_6, l1_cluster_centers, l1_edges, training
        )
        t_features_6 = t_features_6 + t_features_2
        t_features_7 = self.layer7(
            t_features_6, l1_labels, l1_cluster_centers, points, training
        )
        final_features = self.classifier(t_features_7)
        return final_features

class Mini_pointgnn_v12(tf.keras.Model):
    '''
    No l0edges, 3 levels
    3-1-3
    '''
    def __init__(self, num_classes, *args, **kwargs):
        super(Mini_pointgnn_v12, self).__init__(*args, **kwargs)
        self.layer1 = gnn.FFN(aggregate_fn=gnn.graph_scatter_sum_fn,
                                output_fn=gnn.MLP_64_64()) 
        self.layer2 = gnn.GNN(aggregate_fn=gnn.graph_scatter_max_fn,
                                edge_feature_fn=gnn.MLP_64_64(),
                                output_fn=gnn.MLP_64_64())
        self.layer2_1 = gnn.GNN(aggregate_fn=gnn.graph_scatter_max_fn,
                                edge_feature_fn=gnn.MLP_64_64(),
                                output_fn=gnn.MLP_64_64()) 
        self.layer2_2 = gnn.GNN(aggregate_fn=gnn.graph_scatter_max_fn,
                                edge_feature_fn=gnn.MLP_64_64(),
                                output_fn=gnn.MLP_64_64()) 
        self.layer3 = gnn.Mini_to_Large()
        self.layer4 = gnn.GNN(aggregate_fn=gnn.graph_scatter_max_fn)
        self.layer5 = gnn.Large_to_Mini()
        self.layer6 = gnn.GNN(aggregate_fn=gnn.graph_scatter_max_fn,
                                edge_feature_fn=gnn.MLP_64_64(),
                                output_fn=gnn.MLP_64_64()) 
        self.layer6_1 = gnn.GNN(aggregate_fn=gnn.graph_scatter_max_fn,
                                edge_feature_fn=gnn.MLP_64_64(),
                                output_fn=gnn.MLP_64_64())
        self.layer6_2 = gnn.GNN(aggregate_fn=gnn.graph_scatter_max_fn,
                                edge_feature_fn=gnn.MLP_64_64(),
                                output_fn=gnn.MLP_64_64())
        self.layer7 = gnn.FBN() 
        self.classifier = gnn.Classifier(num_classes)

    def call(self, inputs, training=False):
        remission, points, l1_cluster_centers, l2_cluster_centers, \
           l1_edges, l2_edges, l1_labels, l2_labels = inputs
        t_features_1 = self.layer1(
            remission, l1_labels, l1_cluster_centers, points
        )
        
        t_features_2 = self.layer2(
            t_features_1, l1_cluster_centers, l1_edges, training
        )
        t_features_2_1 = self.layer2_1(
            t_features_2, l1_cluster_centers, l1_edges, training
        )
        t_features_2_2 = self.layer2_2(
            t_features_2_1, l1_cluster_centers, l1_edges, training
        )
        t_features_3 = self.layer3(
            t_features_2_2, l2_labels, l2_cluster_centers, l1_cluster_centers, training
        )
        t_features_4 = self.layer4(
            t_features_3, l2_cluster_centers, l2_edges, training
        )
        t_features_5 = self.layer5(
            t_features_4, l2_labels, l2_cluster_centers, l1_cluster_centers, training
        )
        t_features_6 = self.layer6(
            t_features_5, l1_cluster_centers, l1_edges, training
        )
        t_features_6 = t_features_6 + t_features_2_2
        t_features_6 = self.layer6_1(
            t_features_6, l1_cluster_centers, l1_edges, training
        )
        t_features_6 = t_features_6 + t_features_2_1
        t_features_6 = self.layer6_2(
            t_features_6, l1_cluster_centers, l1_edges, training
        )
        t_features_6 = t_features_6 + t_features_2
        t_features_7 = self.layer7(
            t_features_6, l1_labels, l1_cluster_centers, points, training
        )
        final_features = self.classifier(t_features_7)
        return final_features

class Mini_pointgnn_v13(tf.keras.Model):
    '''
    No l0edges, 3 levels
    0-2-0
    '''
    def __init__(self, *args, **kwargs):
        super(Mini_pointgnn_v13, self).__init__(*args, **kwargs)
        self.layer1 = gnn.FFN(aggregate_fn=gnn.graph_scatter_sum_fn,
                                output_fn=gnn.MLP_64_64()) 
        # self.layer2 = gnn.GNN(aggregate_fn=gnn.graph_scatter_max_fn,
        #                         edge_feature_fn=gnn.MLP_64_64(),
        #                         output_fn=gnn.MLP_64_64()) 
        self.layer3 = gnn.Mini_to_Large()
        self.layer4 = gnn.GNN(aggregate_fn=gnn.graph_scatter_max_fn)
        self.layer4_1 = gnn.GNN(aggregate_fn=gnn.graph_scatter_max_fn)
        self.layer5 = gnn.Large_to_Mini()
        # self.layer6 = gnn.GNN(aggregate_fn=gnn.graph_scatter_max_fn,
        #                         edge_feature_fn=gnn.MLP_64_64(),
        #                         output_fn=gnn.MLP_64_64()) 
        self.layer7 = gnn.FBN() 
        self.classifier = gnn.Classifier()

    def call(self, inputs, training=False):
        remission, points, l1_cluster_centers, l2_cluster_centers, \
           l1_edges, l2_edges, l1_labels, l2_labels = inputs
        t_features_1 = self.layer1(
            remission, l1_labels, l1_cluster_centers, points
        )
        
        # t_features_2 = self.layer2(
        #     t_features_1, l1_cluster_centers, l1_edges, training
        # )
        # t_features_2 = self.layer2_1(
        #     t_features_2, l1_cluster_centers, l1_edges, training
        # )
        t_features_3 = self.layer3(
            t_features_1, l2_labels, l2_cluster_centers, l1_cluster_centers, training
        )
        t_features_4 = self.layer4(
            t_features_3, l2_cluster_centers, l2_edges, training
        )
        t_features_4 = self.layer4_1(
            t_features_4, l2_cluster_centers, l2_edges, training
        )
        t_features_5 = self.layer5(
            t_features_4, l2_labels, l2_cluster_centers, l1_cluster_centers, training
        )
        
        t_features_7 = self.layer7(
            t_features_5, l1_labels, l1_cluster_centers, points, training
        )
        final_features = self.classifier(t_features_7)
        return final_features

class Mini_pointgnn_v14(tf.keras.Model):
    '''
    No l0edges, 3 levels
    2-2-2
    '''
    def __init__(self, num_classes, *args, **kwargs):
        super(Mini_pointgnn_v14, self).__init__(*args, **kwargs)
        self.layer1 = gnn.FFN(aggregate_fn=gnn.graph_scatter_sum_fn,
                                output_fn=gnn.MLP_64_64()) 
        self.layer2 = gnn.GNN(aggregate_fn=gnn.graph_scatter_max_fn,
                                edge_feature_fn=gnn.MLP_64_64(),
                                output_fn=gnn.MLP_64_64())
        self.layer2_1 = gnn.GNN(aggregate_fn=gnn.graph_scatter_max_fn,
                                edge_feature_fn=gnn.MLP_64_64(),
                                output_fn=gnn.MLP_64_64()) 
        self.layer3 = gnn.Mini_to_Large()
        self.layer4 = gnn.GNN(aggregate_fn=gnn.graph_scatter_max_fn)
        self.layer4_1 = gnn.GNN(aggregate_fn=gnn.graph_scatter_max_fn)
        self.layer5 = gnn.Large_to_Mini()
        self.layer6 = gnn.GNN(aggregate_fn=gnn.graph_scatter_max_fn,
                                edge_feature_fn=gnn.MLP_64_64(),
                                output_fn=gnn.MLP_64_64()) 
        self.layer6_1 = gnn.GNN(aggregate_fn=gnn.graph_scatter_max_fn,
                                edge_feature_fn=gnn.MLP_64_64(),
                                output_fn=gnn.MLP_64_64())
        self.layer7 = gnn.FBN() 
        self.classifier = gnn.Classifier(num_classes)

    def call(self, inputs, training=False):
        remission, points, l1_cluster_centers, l2_cluster_centers, \
           l1_edges, l2_edges, l1_labels, l2_labels = inputs
        t_features_1 = self.layer1(
            remission, l1_labels, l1_cluster_centers, points
        )
        
        t_features_2 = self.layer2(
            t_features_1, l1_cluster_centers, l1_edges, training
        )
        t_features_2_1 = self.layer2_1(
            t_features_2, l1_cluster_centers, l1_edges, training
        )
        t_features_3 = self.layer3(
            t_features_2_1, l2_labels, l2_cluster_centers, l1_cluster_centers, training
        )
        t_features_4 = self.layer4(
            t_features_3, l2_cluster_centers, l2_edges, training
        )
        t_features_4_1 = self.layer4(
            t_features_4, l2_cluster_centers, l2_edges, training
        )
        t_features_5 = self.layer5(
            t_features_4_1, l2_labels, l2_cluster_centers, l1_cluster_centers, training
        )
        t_features_6 = self.layer6(
            t_features_5, l1_cluster_centers, l1_edges, training
        )
        t_features_6 = t_features_6 + t_features_2_1
        t_features_6 = self.layer6_1(
            t_features_6, l1_cluster_centers, l1_edges, training
        )
        t_features_6 = t_features_6 + t_features_2
        t_features_7 = self.layer7(
            t_features_6, l1_labels, l1_cluster_centers, points, training
        )
        final_features = self.classifier(t_features_7)
        return final_features

class Mini_pointgnn_v15(tf.keras.Model):
    '''
    No l0edges, 3 levels
    4-1-4
    '''
    def __init__(self, *args, **kwargs):
        super(Mini_pointgnn_v15, self).__init__(*args, **kwargs)
        self.layer1 = gnn.FFN(aggregate_fn=gnn.graph_scatter_sum_fn,
                                output_fn=gnn.MLP_64_64()) 
        self.layer2 = gnn.GNN(aggregate_fn=gnn.graph_scatter_max_fn,
                                edge_feature_fn=gnn.MLP_64_64(),
                                output_fn=gnn.MLP_64_64())
        self.layer2_1 = gnn.GNN(aggregate_fn=gnn.graph_scatter_max_fn,
                                edge_feature_fn=gnn.MLP_64_64(),
                                output_fn=gnn.MLP_64_64()) 
        self.layer2_2 = gnn.GNN(aggregate_fn=gnn.graph_scatter_max_fn,
                                edge_feature_fn=gnn.MLP_64_64(),
                                output_fn=gnn.MLP_64_64())
        self.layer2_3 = gnn.GNN(aggregate_fn=gnn.graph_scatter_max_fn,
                                edge_feature_fn=gnn.MLP_64_64(),
                                output_fn=gnn.MLP_64_64()) 
        self.layer3 = gnn.Mini_to_Large()
        self.layer4 = gnn.GNN(aggregate_fn=gnn.graph_scatter_max_fn)
        # self.layer4_1 = gnn.GNN(aggregate_fn=gnn.graph_scatter_max_fn)
        self.layer5 = gnn.Large_to_Mini()
        self.layer6 = gnn.GNN(aggregate_fn=gnn.graph_scatter_max_fn,
                                edge_feature_fn=gnn.MLP_64_64(),
                                output_fn=gnn.MLP_64_64()) 
        self.layer6_1 = gnn.GNN(aggregate_fn=gnn.graph_scatter_max_fn,
                                edge_feature_fn=gnn.MLP_64_64(),
                                output_fn=gnn.MLP_64_64())
        self.layer6_2 = gnn.GNN(aggregate_fn=gnn.graph_scatter_max_fn,
                                edge_feature_fn=gnn.MLP_64_64(),
                                output_fn=gnn.MLP_64_64())
        self.layer6_3 = gnn.GNN(aggregate_fn=gnn.graph_scatter_max_fn,
                                edge_feature_fn=gnn.MLP_64_64(),
                                output_fn=gnn.MLP_64_64())
        self.layer7 = gnn.FBN() 
        self.classifier = gnn.Classifier()

    def call(self, inputs, training=False):
        remission, points, l1_cluster_centers, l2_cluster_centers, \
           l1_edges, l2_edges, l1_labels, l2_labels = inputs
        t_features_1 = self.layer1(
            remission, l1_labels, l1_cluster_centers, points
        )
        
        t_features_2 = self.layer2(
            t_features_1, l1_cluster_centers, l1_edges, training
        )
        t_features_2_1 = self.layer2_1(
            t_features_2, l1_cluster_centers, l1_edges, training
        )
        t_features_2_2 = self.layer2_2(
            t_features_2_1, l1_cluster_centers, l1_edges, training
        )
        t_features_2_3 = self.layer2_3(
            t_features_2_2, l1_cluster_centers, l1_edges, training
        )
        t_features_3 = self.layer3(
            t_features_2_3, l2_labels, l2_cluster_centers, l1_cluster_centers, training
        )
        t_features_4 = self.layer4(
            t_features_3, l2_cluster_centers, l2_edges, training
        )
        t_features_5 = self.layer5(
            t_features_4, l2_labels, l2_cluster_centers, l1_cluster_centers, training
        )
        t_features_6 = self.layer6(
            t_features_5, l1_cluster_centers, l1_edges, training
        )
        t_features_6 = t_features_6 + t_features_2_3
        t_features_6 = self.layer6_1(
            t_features_6, l1_cluster_centers, l1_edges, training
        )
        t_features_6 = t_features_6 + t_features_2_2
        t_features_6 = self.layer6_2(
            t_features_6, l1_cluster_centers, l1_edges, training
        )
        t_features_6 = t_features_6 + t_features_2_1
        t_features_6 = self.layer6_3(
            t_features_6, l1_cluster_centers, l1_edges, training
        )
        t_features_6 = t_features_6 + t_features_2
        t_features_7 = self.layer7(
            t_features_6, l1_labels, l1_cluster_centers, points, training
        )
        final_features = self.classifier(t_features_7)
        return final_features
def get_model(model):
    models = {
        'Mini_pointgnn' : Mini_pointgnn,
        'Mini_pointgnn_v1': Mini_pointgnn_v1,
        'Mini_pointgnn_v2': Mini_pointgnn_v2,
        'Mini_pointgnn_v3': Mini_pointgnn_v3,
        'Mini_pointgnn_v4': Mini_pointgnn_v4,
        'Mini_pointgnn_v5': Mini_pointgnn_v5,
        'Mini_pointgnn_v6': Mini_pointgnn_v6,
        'Mini_pointgnn_v7': Mini_pointgnn_v7,
        'Mini_pointgnn_v8': Mini_pointgnn_v8,
        'Mini_pointgnn_v9': Mini_pointgnn_v9,
        'Mini_pointgnn_v10': Mini_pointgnn_v10,
        'Mini_pointgnn_v11': Mini_pointgnn_v11,
        'Mini_pointgnn_v12': Mini_pointgnn_v12,
        'Mini_pointgnn_v13': Mini_pointgnn_v13,
        'Mini_pointgnn_v14': Mini_pointgnn_v14,
        'Mini_pointgnn_v15': Mini_pointgnn_v15,
        'MultiScaleGNN': MultiScaleGNN,
        
    }
    return models[model]
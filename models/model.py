#from typing import final
import tensorflow as tf

import models.gnn as gnn  

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

def get_model(model):
    models = {
        'Mini_pointgnn_v7': Mini_pointgnn_v7,
        'Mini_pointgnn_v12': Mini_pointgnn_v12
        
    }
    return models[model]
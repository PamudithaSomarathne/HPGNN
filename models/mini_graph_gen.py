import tensorflow as tf
import numpy as np
from sklearn.neighbors import NearestNeighbors

def cylindrical_partition(points,
                        rsq_parts,
                        theta_parts):
    rsq, theta, phi = (np.linalg.norm(points[:,:2], axis=1))/rsq_parts,\
                        np.arctan2(points[:,1], points[:,0])/theta_parts,\
                        points[:,2] /0.5
    rtp = np.array([rsq,theta,phi], dtype=np.int32).T   # Stack properly
    # rtp = np.stack([rsq,theta,phi], axis=0)
    centers, labels = np.unique(rtp, axis=0, return_inverse=True)
    rng = np.random.default_rng()
    sources = np.array([], dtype=np.uint32)
    dests = np.array([], dtype=np.uint32)
    cluster_centers = np.array([], dtype=np.uint32)
    
    for i in range(centers.shape[0]):
        sources_i = np.where(labels==i)[0]
        sources =np.append(sources, sources_i)
        rng.shuffle(sources_i)
        dests = np.append(dests, sources_i) #dests
    cluster_centers = tf.math.unsorted_segment_mean(points, labels, centers.shape[0])
    edges = np.stack([sources, dests], axis=1)
    # indices = np.argsort(edges[:,1])
    edges = np.array(edges, dtype=np.int32)
    cluster_centers = np.array(cluster_centers, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    return cluster_centers, labels, edges

def cylindrical_partition_nol0(points,
                        rsq_parts,
                        theta_parts,
                        z_parts = 0.5):
    rsq, theta, phi = (np.linalg.norm(points[:,:2], axis=1))/rsq_parts,\
                        np.arctan2(points[:,1], points[:,0])/theta_parts,\
                        points[:,2] / z_parts
    rtp = np.array([rsq,theta,phi], dtype=np.int32).T   # Stack properly
    # rtp = np.stack([rsq,theta,phi], axis=0)
    centers, labels = np.unique(rtp, axis=0, return_inverse=True)
    cluster_centers = tf.math.unsorted_segment_mean(points, labels, centers.shape[0])
    cluster_centers = np.array(cluster_centers, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    return cluster_centers, labels

def make_l1_edges(cluster_centers, radius=4, num_neighbors= 256, limit_neighbors=True):
    nbrs = NearestNeighbors(radius=radius, algorithm='ball_tree').fit(cluster_centers).fit(
        cluster_centers
    )
    indices = nbrs.radius_neighbors(cluster_centers, return_distance=False)

    if limit_neighbors:
        indices = [neighbors if neighbors.size <= num_neighbors else
                np.random.choice(neighbors, num_neighbors, replace=False)
                for neighbors in indices]
    vertices_v = np.concatenate(indices)
    vertices_i = np.concatenate(
        [i*np.ones(neighbors.size, dtype=np.int32)
            for i, neighbors in enumerate(indices)])
    vertices = np.array([vertices_v, vertices_i], dtype=np.int32).transpose().reshape(-1,2)
    vertices = np.array(vertices, dtype=np.int32)
    return vertices

def graph_maker_cylinder(points, remission):
    rst_part = 2
    theta_part = 2 * np.pi / 32
    cluster_centers, labels, edges_l0 = cylindrical_partition(points, rst_part, theta_part)
    edges_l1 = make_l1_edges(cluster_centers, radius=4, 
                                num_neighbors= 8, limit_neighbors=True)
    return edges_l1, edges_l0, cluster_centers, labels, points, remission

def graph_maker_cylinder_3levels(points, remission):
    l1_rst_part = 0.5
    l1_theta_part = 2 * np.pi / 64
    l1_z_part = 0.25
    l2_rst_part = 2
    l2_theta_part = 2 * np.pi / 32
    l2_z_part = 1.5
    l1_cluster_centers, l1_labels = cylindrical_partition_nol0(points, l1_rst_part, l1_theta_part, l1_z_part)
    l2_cluster_centers, l2_labels = cylindrical_partition_nol0(l1_cluster_centers, l2_rst_part, l2_theta_part, l2_z_part)
    edges_l1 = make_l1_edges(l1_cluster_centers, radius=1, 
                                num_neighbors= 16, limit_neighbors=True)
    edges_l2 = make_l1_edges(l2_cluster_centers, radius = 4, num_neighbors=16, limit_neighbors=True)
    return edges_l1, edges_l2, l1_cluster_centers, l2_cluster_centers, l1_labels, l2_labels, points, remission


def graph_maker_map_cylinder(points, remission, sem_label):
    edges_l1, edges_l0, cluster_centers, labels, points, \
        remission = tf.numpy_function(
            graph_maker_cylinder, [points, remission], 
            [tf.int32, tf.int32, tf.float32, tf.int32, tf.float32, tf.float32], name = 'graph_maker_cylinder'
        )
    return edges_l1, edges_l0, cluster_centers, labels, points, remission, sem_label

def graph_maker_map_cylinder_3levels(points, remission, sem_label):
    edges_l1, edges_l2, l1_cluster_centers, l2_cluster_centers, l1_labels,\
        l2_labels, points, remission = tf.numpy_function(
            graph_maker_cylinder_3levels, [points, remission], 
            [tf.int32, tf.int32, tf.float32, tf.float32, tf.int32,  tf.int32, tf.float32, tf.float32]
        )
    return edges_l1, edges_l2, l1_cluster_centers, l2_cluster_centers, l1_labels, l2_labels, points, remission, sem_label


def get_graph_maker(func):
    functions = {
        'graph_maker_map_cylinder': graph_maker_map_cylinder,
        'graph_maker_map_cylinder_3levels': graph_maker_map_cylinder_3levels
    }
    return functions[func]


if __name__ == "__main__":
    # points = np.array([[1,1,1],[0,0,1],[0,1,1],[1,0,1],[0.5,0,1],[0.7,0,1]])
    # print(spherical_partition(points, 1, 1))
    # labels = np.array([0,0,0,0,0,0,0,0,0,0,1,1,1,1,1])
    # edges_l0 = make_2N_l0_edges_tf(labels, 2)
    # print(edges_l0)
    pass

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
    
class dataset_nuscenes():
    def __init__(self, dataset_path = '', velo_split = '', label_split='', voxel_size = 0.5,
                    rotate_aug = False,
                    flip_aug = False,
                    scale_aug = False):
        self.velo_data_path = ''
        self.labels_path = ''
        self.velo_split = velo_split
        self.label_split = label_split
        self.dataset_path = dataset_path
        self.base_voxel_size = voxel_size
        self.keys = tf.constant([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31], dtype=tf.int32)
        self.values = tf.constant([0,0,7,7,7,0,7,0,0,1,0,0,8,0,2,3,3,4,5,0,0,6,9,10,11,12,13,14,15,0,16,0], dtype=tf.int32)
        self.seg_map = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(self.keys, self.values), self.values[0], name='seg_map')
        self.rotate_aug = rotate_aug
        self.flip_aug = flip_aug
        self.scale_aug = scale_aug

    def read_velo(self, filename):
        path_to_file = self.dataset_path + 'velo/' + filename.numpy()[0].decode('utf-8')
        velo = np.fromfile(path_to_file, dtype=np.float32).reshape(-1,5)
        points = velo[:,0:3]
        remission = velo[:,2:4]
        return points, remission

    def read_label(self, filename):
        label_file = self.dataset_path + 'lidarseg/' + filename.numpy()[0].decode('utf-8')
        label = np.fromfile(label_file, dtype= np.uint8).astype(np.int32)
        return label.reshape(-1,1)
        
    def data_read(self, velo_names, label_names):
        points, remission = tf.py_function(self.read_velo, [velo_names], [tf.float32, tf.float32], name='data_read_velo')
        sem_label = tf.py_function(self.read_label, [label_names], [tf.int32], name='data_read_label')

        return (points, remission, sem_label)

    def read_filenames(self):
        with open(self.velo_split, 'r') as f:
            velo_names = f.read().splitlines()
        with open(self.label_split, 'r') as f:
            label_names = f.read().splitlines()
        return (velo_names, label_names)
    
    def process_labels(self, points, remission, seg_labels):
        mapped_seg = tf.reshape(tf.one_hot(self.seg_map.lookup(seg_labels), 17, dtype=tf.float32), (-1, 17))
        return (points, remission, mapped_seg)

    def augmentation(self, points, remission):
        aug = np.random.choice(3,1)
        if self.rotate_aug and aug == 0:
            rotate_rad = np.deg2rad(np.random.random() * 360)
            c, s = np.cos(rotate_rad), np.sin(rotate_rad)
            j = np.matrix([[c, s], [-s, c]])
            points[:, :2] = np.dot(points[:, :2], j)
        if self.flip_aug and aug == 1:
            flip_type = np.random.choice(4, 1)
            if flip_type == 1:
                points[:, 0] = -points[:, 0]
            elif flip_type == 2:
                points[:, 1] = -points[:, 1]
            elif flip_type == 3:
                points[:, :2] = -points[:, :2]
        if self.scale_aug and aug == 2:
            noise_scale = np.random.uniform(0.95, 1.05)
            points[:, 0] = noise_scale * points[:, 0]
            points[:, 1] = noise_scale * points[:, 1]
        return points, remission
   
    def augmentation_map(self, points, remission, seg_labels):
        points, remission = tf.numpy_function(
            self.augmentation, [points, remission],
            [tf.float32, tf.float32], name = 'augmentation'
        )
        return points, remission, seg_labels

class dataset_kitti():
    def __init__(self, dataset_path = '', velo_split = '', label_split='', voxel_size = 0.5,
                    rotate_aug = False,
                    flip_aug = False,
                    scale_aug = False):
        self.velo_data_path = ''
        self.labels_path = ''
        self.velo_split = velo_split
        self.label_split = label_split
        self.dataset_path = dataset_path
        self.base_voxel_size = voxel_size
        self.keys = tf.constant([0, 1, 10, 11, 13, 15, 16, 18, 20, 30, 31, 32, 40, 44, 48, 49, 50, 51, 52, 60, 70, 71, 72, 80, 81, 99, 252, 253, 254, 255, 256, 257, 258, 259], dtype=tf.int32)
        self.values = tf.constant([0, 0, 1, 2, 5, 3, 5, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 9, 15, 16, 17, 18, 19, 0, 1, 7, 6, 8, 5, 5, 4, 5], dtype=tf.int32)
        self.seg_map = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(self.keys, self.values), self.values[0], name='seg_map')
        self.rotate_aug = rotate_aug
        self.flip_aug = flip_aug
        self.scale_aug = scale_aug

    def read_velo(self, filename):
        path_to_file = self.dataset_path + filename.numpy()[0].decode('utf-8') + '.bin'
        velo = np.fromfile(path_to_file, dtype=np.float32).reshape(-1,4)
        points = velo[:,0:3]
        remission = velo[:,3].reshape(-1,1)
        return points, remission

    def read_label(self, filename):
        label_file = self.dataset_path + filename.numpy()[0].decode('utf-8') + '.label'
        label = np.fromfile(label_file, dtype= np.uint32)
        label = label.reshape(-1)
        sem_label = label & 0xFFFF
        return sem_label.reshape(-1,1)
        
    def data_read(self, velo_names, label_names):
        points, remission = tf.py_function(self.read_velo, [velo_names], [tf.float32, tf.float32])
        sem_label = tf.py_function(self.read_label, [label_names], [tf.int32])
        return (points, remission, sem_label)
    
    def read_filenames(self):
        with open(self.velo_split, 'r') as f:
            velo_names = f.read().splitlines()
        with open(self.label_split, 'r') as f:
            label_names = f.read().splitlines()
        return (velo_names, label_names)
    
    def process_labels(self, points, remission, seg_labels):
        mapped_seg = tf.reshape(tf.one_hot(self.seg_map.lookup(seg_labels), 20, dtype=tf.float32), (-1, 20))
        return (points, remission, mapped_seg)

    def augmentation(self, points, remission, seg_labels):
        aug = np.random.choice(3,1)
        if self.rotate_aug and aug == 0:
            rotate_rad = np.deg2rad(np.random.random() * 360)
            c, s = np.cos(rotate_rad), np.sin(rotate_rad)
            j = np.matrix([[c, s], [-s, c]])
            points[:, :2] = np.dot(points[:, :2], j)
        if self.flip_aug and aug == 1:
            flip_type = np.random.choice(4, 1)
            if flip_type == 1:
                points[:, 0] = -points[:, 0]
            elif flip_type == 2:
                points[:, 1] = -points[:, 1]
            elif flip_type == 3:
                points[:, :2] = -points[:, :2]
        if self.scale_aug and aug == 2:
            noise_scale = np.random.uniform(0.95, 1.05)
            points[:, 0] = noise_scale * points[:, 0]
            points[:, 1] = noise_scale * points[:, 1]
        return points, remission, seg_labels
   
    def augmentation_map(self, points, remission, seg_labels):
        points, remission, seg_labels = tf.numpy_function(
            self.augmentation, [points, remission, seg_labels],
            [tf.float32, tf.float32, tf.float32]
        )
        return points, remission, seg_labels
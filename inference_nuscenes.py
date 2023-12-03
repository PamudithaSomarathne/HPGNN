import os
import datetime
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.config.run_functions_eagerly(True)
import numpy as np
import argparse
from tqdm import tqdm
import json
from models.mini_graph_gen import get_graph_maker
from models.model import get_model

parser = argparse.ArgumentParser(description='FEPODLOML Training')
parser.add_argument('--config', type=str, required=True,
                    help='Path to model configs')
args = parser.parse_args()

CONFIG_PATH = 'configs/' + args.config
with open(CONFIG_PATH, 'r') as f:
    CONFIG = json.load(f)

MODEL_PATH = 'checkpoints/' +  CONFIG["MODEL_NAME"] + '.' + str(CONFIG["REVISION"]) + '/'
# data reading
DATASET_DIR = 'dataset/nuscenes/'
SPLIT_FILE = 'splits/nuscenes_velo_test.txt'
LABEL_FILE = 'splits/nuscenes_labels_test.txt'

def read_velo(filename):
    path_to_file = DATASET_DIR + 'velo/' + filename
    velo = np.fromfile(path_to_file, dtype=np.float32).reshape(-1,5)
    points = velo[:,0:3]
    remission = velo[:,2:4]
    return points, remission

def data_read(velo_names):
    points, remission = tf.py_function(read_velo, [velo_names], [tf.float32, tf.float32])
    return (points, remission)

@tf.function(experimental_relax_shapes=True)
def forward(inputs):
    output_features = tf.argmax(segmenter(inputs), axis=1)
    return output_features

graph_fn = get_graph_maker(CONFIG["GRAPH_FN"])
segmenter = get_model(CONFIG["MODEL_TYPE"])(CONFIG["NUM_CLASSES"])

inv_label = np.array([1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], dtype=np.uint8)

if __name__ == '__main__':
    with open(SPLIT_FILE, 'r') as f:
        file_names = f.read().splitlines()
    
    with open(LABEL_FILE, 'r') as f:
        label_names = f.read().splitlines()
    
    assert len(label_names) == len(file_names)

    num_samples = len(file_names)
        
    segmenter.load_weights(MODEL_PATH)
    print("Model loaded from checkpoint")
    print("Timestamp:", datetime.datetime.fromtimestamp(os.stat(MODEL_PATH+'checkpoint').st_mtime))

    for index in tqdm(range(num_samples)):
        file = file_names[index]
        lab = label_names[index]
        points, remission = read_velo(file)
        edges_l1, edges_l2, l1_cluster_centers, l2_cluster_centers, l1_labels, l2_labels,\
                points, remission, sem_label = graph_fn(
                                            points, remission, 1
                                        )
        inputs = (remission, points, l1_cluster_centers, 
                    l2_cluster_centers, edges_l1, edges_l2, 
                    l1_labels, l2_labels)
        output_features = forward(inputs).numpy()
        output_features = inv_label[output_features]
        output_features.tofile(open(DATASET_DIR + 'lidarseg/test/' + lab, 'wb'))
    print('Inference finished for', MODEL_PATH)
        

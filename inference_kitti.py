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
SPLIT_FILE = 'splits/kitti_velo_test.txt'
DATASET_DIR = 'dataset/sequences/'

def read_velo(filename):
    path_to_file = DATASET_DIR + filename + '.bin'
    velo = np.fromfile(path_to_file, dtype=np.float32).reshape(-1,4)
    points = velo[:,0:3]
    remission = velo[:,3].reshape(-1,1)
    return points, remission

@tf.function(experimental_relax_shapes=True)
def forward(inputs):
    output_features = tf.argmax(segmenter(inputs), axis=1)
    return output_features
inv_label = np.array([0, 10, 11, 15, 18, 20, 30, 31, 32, 40, 44, 48, 49, 50, 51, 70, 71, 72, 80, 81], dtype=np.uint32)

graph_fn = get_graph_maker(CONFIG["GRAPH_FN"])
segmenter = get_model(CONFIG["MODEL_TYPE"])(CONFIG["NUM_CLASSES"])
if __name__ == '__main__':
    with open(SPLIT_FILE, 'r') as f:
        file_names = f.read().splitlines()

    segmenter.load_weights(MODEL_PATH)
    print("Model loaded from checkpoint")
    print("Timestamp:", datetime.datetime.fromtimestamp(os.stat(MODEL_PATH+'checkpoint').st_mtime))

    for file in tqdm(file_names):
        points, remission = read_velo(file)
        l1_edges, l2_edges, l1_cluster_centers, l2_cluster_centers, \
                l1_labels, l2_labels, points, remission, sem_label = graph_fn(
                                            points, remission, 1)
        inputs = (remission, points, l1_cluster_centers, 
                    l2_cluster_centers, l1_edges, l2_edges, 
                    l1_labels, l2_labels)
        output_features = forward(inputs)
        output_features = output_features.numpy().astype(np.uint32)
        output_features = inv_label[output_features]
        os.makedirs(DATASET_DIR + 'predictions/' + file[:-7], exist_ok=True)
        output_features.tofile(open(DATASET_DIR + 'predictions/' + file + '.label', 'wb'))
    print('Inference finished for', MODEL_PATH)
        

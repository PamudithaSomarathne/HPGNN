import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
tf.config.run_functions_eagerly(True)
import time
import datetime
import numpy as np
import argparse
import json

from models.data_pipeline import dataset_kitti as dataset
from models.model import get_model

from models.mini_graph_gen import get_graph_maker
from models.loss import get_loss_fn

parser = argparse.ArgumentParser(description='FEPODLOML Training')
parser.add_argument('--config', type=str, required=True,
                    help='Path to config file')
parser.add_argument('--dataset', type=str, required=False,
                    default='dataset/sequences/',
                    help='Dataset directory')
parser.add_argument('--debug', type=bool, required=False,
                    default=False, help='Need debugging?')
args = parser.parse_args()

CONFIG_PATH = 'configs/' + args.config
with open(CONFIG_PATH, 'r') as f:
    CONFIG = json.load(f)

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

@tf.function(experimental_relax_shapes=True)
def train_step(remission, 
                points, 
                l1_cluster_centers,
                l2_cluster_centers,
                edges_l1, 
                edges_l2,
                l1_labels,
                l2_labels,
                sem_label):
    with tf.GradientTape(persistent=True) as tape:
        output_features = segmenter((remission, points, l1_cluster_centers, l2_cluster_centers, \
           edges_l1, edges_l2, l1_labels, l2_labels), training=False) + 1e-8
        l_loss = compute_lovasz_loss(sem_label, output_features)
        wce_loss = compute_loss(sem_label, output_features) 
        reg_loss = GAMMA * tf.reduce_sum(segmenter.losses)
        loss = l_loss + reg_loss + wce_loss
    grads = tape.gradient(loss, segmenter.trainable_variables)
    optimizer.apply_gradients(zip(grads, segmenter.trainable_variables))
    iou.update_state(tf.argmax(sem_label, axis=1), tf.argmax(output_features, axis=1))
    acc.update_state(tf.argmax(sem_label, axis=1), tf.argmax(output_features, axis=1))
    return loss, l_loss, reg_loss, wce_loss

@tf.function(experimental_relax_shapes=True)
def val_step(remission, points, l1_cluster_centers, l2_cluster_centers, edges_l1, edges_l2,\
                            l1_labels, l2_labels, sem_label):
    output_features = segmenter((remission, points, l1_cluster_centers, l2_cluster_centers, \
           edges_l1, edges_l2, l1_labels, l2_labels), training=False)
    val_iou.update_state(tf.argmax(sem_label, axis=1), tf.argmax(output_features, axis=1))
    val_acc.update_state(tf.argmax(sem_label, axis=1), tf.argmax(output_features, axis=1))
    
def train_segmenter(dataset, val_dset,
                    TRAINING_CONFIG,
                    GRAPH_CONFIG, MODEL, REVISION, DEBUG):
    '''A function to train the segmentation model'''
    if TRAINING_CONFIG["RECORD_LOGS"]:
        train_log_dir = TRAINING_CONFIG["LOGS_PATH"]  + MODEL + '.' + str(REVISION) + '/' 
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        
    if TRAINING_CONFIG["LOAD_MODEL"]:
        segmenter.load_weights(TRAINING_CONFIG["WEIGHTS_PATH"] + MODEL + '.' + str(REVISION) + '/') 
        print("loaded from checkpoint")
    for epoch in range(1,TRAINING_CONFIG["EPOCHS"]+1):
        for i,d in enumerate(dataset):
            edges_l1, edges_l2, l1_cluster_centers, l2_cluster_centers, l1_labels, l2_labels,\
                points, remission, sem_label = d

            loss, lovasz_loss, reg_loss, wce_loss = train_step(remission, 
                                                    points, 
                                                    l1_cluster_centers,
                                                    l2_cluster_centers,
                                                    edges_l1, 
                                                    edges_l2,
                                                    l1_labels,
                                                    l2_labels,
                                                    sem_label)

            if TRAINING_CONFIG["VERBOSE"]:
                print('step: {0} \t | Total loss: {1:.3f} \t | Lovasz loss: {2:.3f} \t| Reg loss: {3:.3f} \t| WCE loss: {4:.3f} \t| mIoU: {5:.3f} \t | acc: {6:.3f}'.format(
                    (epoch-1)*len(velo_names)+i, 
                    loss.numpy(), 
                    lovasz_loss.numpy(), 
                    reg_loss.numpy(), 
                    wce_loss.numpy(), 
                    iou.result().numpy(), 
                    acc.result().numpy()))

            if TRAINING_CONFIG["RECORD_LOGS"]:
                step=(epoch-1)*len(velo_names)//BATCH_SIZE+i
                with train_summary_writer.as_default():
                    tf.summary.scalar('Total loss', loss, step=step )
                    tf.summary.scalar('Regularization loss', reg_loss, step=step )
                    tf.summary.scalar('Lovasz loss', lovasz_loss, step=step )
                    tf.summary.scalar(CONFIG["LOSS"]["FN"], wce_loss, step=step )
                    tf.summary.scalar('mIoU', iou.result().numpy(), step=step )
                    tf.summary.scalar('Accuracy', acc.result().numpy(), step=step )
                    tf.summary.scalar('Learning Rate', lr_schedule(step), step=step )
                    train_summary_writer.flush()
            if DEBUG:
                break
        
        if TRAINING_CONFIG["SAVE_WEIGHTS"]:
            segmenter.save_weights(TRAINING_CONFIG["WEIGHTS_PATH"] + MODEL + '.' + str(REVISION) + '/') 
            print('epoch: {0}    saved weigths at {1} of {2}.{3}'.format(epoch, datetime.datetime.now(), MODEL, REVISION))
        iou.reset_states()
        acc.reset_state()

        if TRAINING_CONFIG["VAL_N_EPOCHS"] > 0 and epoch % TRAINING_CONFIG["VAL_N_EPOCHS"] == 0:
            print('Running Evaluation on epoch {}'.format(epoch))
            for d in val_dset:
                edges_l1, edges_l2, l1_cluster_centers, l2_cluster_centers, l1_labels, l2_labels,\
                    points, remission, sem_label = d
                val_step(remission, points, l1_cluster_centers, l2_cluster_centers, edges_l1, edges_l2,\
                            l1_labels, l2_labels, sem_label)
            if TRAINING_CONFIG["RECORD_LOGS"]:
                with train_summary_writer.as_default():
                    tf.summary.scalar('Validation mIoU', val_iou.result().numpy(), step=epoch )
                    tf.summary.scalar('Validation Accuracy', val_acc.result().numpy(), step=epoch )
                    train_summary_writer.flush()
            print('Validation Accuracy: {} \tValidation mIoU: {}'.format(
                val_acc.result().numpy(),
                val_iou.result().numpy()
            ))
            val_iou.reset_states()
            val_acc.reset_state()
        
        print("Finished epoch", epoch, "at", datetime.datetime.now())


if __name__ == '__main__':
    # model
    segmenter = get_model(CONFIG["MODEL_TYPE"])(CONFIG["NUM_CLASSES"])
    graph_fn = get_graph_maker(CONFIG["GRAPH_FN"])
    OPTIMIZER_CONFIG = CONFIG["OPTIMIZER"]

    # metrics, losses
    iou = tf.keras.metrics.MeanIoU(num_classes=CONFIG["NUM_CLASSES"], name='iou')
    acc = tf.keras.metrics.Accuracy(name='accuracy')
    val_iou = tf.keras.metrics.MeanIoU(num_classes=CONFIG["NUM_CLASSES"], name='val_iou')
    val_acc = tf.keras.metrics.Accuracy(name='val_accuracy')
    lovasz_loss = get_loss_fn('Lovasz_loss')(num_classes=CONFIG["NUM_CLASSES"])
    loss_fn = get_loss_fn(CONFIG["LOSS"]["FN"])(loss_kwargs=CONFIG["LOSS"])
    def compute_lovasz_loss(labels, predictions):
        loss = lovasz_loss(labels, predictions) 
        return tf.nn.compute_average_loss(loss, global_batch_size=BATCH_SIZE)
    def compute_loss(labels, predictions):
        per_example_loss = loss_fn(labels, predictions) 
        return tf.nn.compute_average_loss(per_example_loss, global_batch_size=BATCH_SIZE)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                        initial_learning_rate=OPTIMIZER_CONFIG["INITIAL_LEARNING_RATE"],
                        decay_steps=OPTIMIZER_CONFIG["DECAY_STEPS"],
                        decay_rate=OPTIMIZER_CONFIG["DECAY_RATE"],
                        staircase=True)
    
    optimizer =tf.keras.optimizers.SGD(
        learning_rate=lr_schedule, momentum=0.8, nesterov=True, global_clipnorm= 1
    )
    
    DATASET_CONFIG = CONFIG["DATASET"]
    # support class for dataset processing
    d_set = dataset(dataset_path=args.dataset,
                    velo_split = "splits/kitti_velo_train.txt",   
                    label_split= "splits/kitti_labels_train.txt",  
                    rotate_aug=True,
                    flip_aug=True,
                    scale_aug = True)
    val_dset = dataset(dataset_path=args.dataset,
                        velo_split = "splits/kitti_velo_val.txt",   
                        label_split= "splits/kitti_labels_val.txt"
                    )
    GAMMA = 5e-3
    
    # dataset preprocessing and loading
    AUTOTUNE = tf.data.AUTOTUNE
    BATCH_SIZE = DATASET_CONFIG["BATCH_SIZE"]
    velo_names, label_names = d_set.read_filenames()
    BUFFER_SIZE = len(velo_names)
    train_ds = tf.data.Dataset.from_tensor_slices((velo_names, label_names))
    train_ds = train_ds.shuffle(BUFFER_SIZE, reshuffle_each_iteration=True).batch(BATCH_SIZE)
    train_ds = train_ds.map(d_set.data_read, num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.map(d_set.process_labels, num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.map(d_set.augmentation_map, num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.map(graph_fn, num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.prefetch(AUTOTUNE)

    #val data
    val_velo_names, val_label_names = val_dset.read_filenames()
    val_ds = tf.data.Dataset.from_tensor_slices((val_velo_names, val_label_names))
    val_ds = val_ds.batch(BATCH_SIZE)
    val_ds = val_ds.map(val_dset.data_read, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.map(val_dset.process_labels, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.map(graph_fn, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)
    # training function
    t1 = time.time()
    print('Training started for {} epochs at {}'.format(CONFIG["TRAINING"]["EPOCHS"], datetime.datetime.now()))
    train_segmenter(dataset=train_ds, 
                    val_dset = val_ds,
                    TRAINING_CONFIG=CONFIG["TRAINING"],
                    GRAPH_CONFIG=CONFIG["GRAPH_GEN"],
                    MODEL=CONFIG["MODEL_NAME"],
                    REVISION=CONFIG["REVISION"],
                    DEBUG=args.debug)
    t2 = time.time()
    print('time taken for {} epochs: {}'.format(CONFIG["TRAINING"]["EPOCHS"], t2-t1))

   

"""Implements popular losses. """

# Suppress CUDA messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
# from models.lovasz_loss import lovasz_softmax

# def lovasz_grad(gt_sorted):
#     gts = tf.reduce_sum(gt_sorted, axis=1, keepdims=True)
#     intersection = gts - tf.cumsum(gt_sorted, axis=1)
#     union = gts + tf.cumsum(1. - gt_sorted, axis=1)
#     jaccard = 1. - intersection / union
#     jaccard_zero = tf.concat((tf.zeros((20,1)), jaccard[:,:-1]), axis=1)
#     return jaccard - jaccard_zero

class Lovasz_softmax(tf.keras.losses.Loss):
    def __init__(self, reduction = tf.keras.losses.Reduction.NONE, ignore = 0, num_classes=20, name = 'lovasz'):
        super().__init__(reduction=reduction, name=name)
        self.ignore = ignore
        self.num_classes = num_classes
        self.weights = tf.convert_to_tensor((np.arange(self.num_classes)).reshape((self.num_classes,1)), dtype= tf.int32)
    
    def call(self, labels, logits):
        labels, logits = tf.transpose(labels), tf.transpose(logits)
        errors = tf.abs(labels - logits) # (C,N)
        sorted_errors, perm = tf.math.top_k(errors, k=tf.shape(labels)[1]) # Desc sort last dim
        flat_labels = tf.reshape(labels, (-1,1))
        perm = tf.reshape(tf.shape(perm)[1] * self.weights + perm, (-1,1))
        sorted_labels = tf.reshape(tf.gather(flat_labels, perm), tf.shape(labels))
        grad = self.lovasz_grad(sorted_labels)
        loss = tf.reduce_sum(tf.multiply(sorted_errors, tf.stop_gradient(grad)), axis=1, keepdims=True)
        return tf.reduce_mean(loss, keepdims=True)
        
    def lovasz_grad(self, gt_sorted):
        gts = tf.reduce_sum(gt_sorted, axis=1, keepdims=True)
        intersection = gts - tf.cumsum(gt_sorted, axis=1)
        union = gts + tf.cumsum(1. - gt_sorted, axis=1)
        jaccard = 1. - intersection / union
        jaccard_zero = tf.concat((tf.zeros((self.num_classes,1)), jaccard[:,:-1]), axis=1)
        return jaccard - jaccard_zero

class Tversky_loss(tf.keras.losses.Loss):
    def __init__(self, loss_kwargs, reduction=tf.keras.losses.Reduction.NONE, name='tversky'):
        super(Tversky_loss, self).__init__(reduction=reduction, name=name)
        self.alpha = loss_kwargs["ALPHA"]
        self.beta = loss_kwargs["BETA"]
        self.gamma = loss_kwargs["GAMMA"]
    
    def call(self, labels, logits):
        y_true_pos = tf.reshape(labels, (-1,1))
        y_pred_pos = tf.reshape(logits, (-1,1))
        true_pos = tf.math.reduce_sum(tf.multiply(y_true_pos, y_pred_pos))
        false_neg = tf.math.reduce_sum(tf.multiply(y_true_pos, (1-y_pred_pos)))
        false_pos = tf.math.reduce_sum(tf.multiply((1-y_true_pos), y_pred_pos))
        ti = (true_pos + 1e-5)/(true_pos + self.alpha*false_neg + self.beta*false_pos + 1e-5)
        return tf.reshape(tf.math.pow((1 - ti), self.gamma), (-1,1))

class Focal_loss(tf.keras.losses.Loss):
  def __init__(self, loss_kwargs, reduction=tf.keras.losses.Reduction.NONE, name = 'focal_loss'):
    super(Focal_loss, self).__init__(reduction=reduction, name=name)
    self.gamma = loss_kwargs["GAMMA"]
    self.use_invalids = loss_kwargs["USE_INVALIDS"]
    if self.use_invalids: self.lamda = loss_kwargs["LAMDA"]
    else: self.lamda = 0
    self.num_classes = loss_kwargs["NUM_CLASSES"]
    self.ignore_classes = np.identity(self.num_classes, dtype=np.float32)
    self.ignore_classes[0] = np.ones((1, self.num_classes), dtype=np.float32)
    self.ignore_classes = tf.convert_to_tensor(self.ignore_classes, dtype=tf.float32)

  def call(self, labels, logits):
    """
        loss = (1 - y_hat_for_true_class)^g * -log(y_hat_for_true_class)
    """
    labels = tf.matmul(labels, self.ignore_classes)
    valid = tf.math.reduce_sum(tf.math.multiply(logits, labels), axis=-1) # Output for true class
    if self.use_invalids: invalid = tf.math.reduce_sum(tf.math.multiply(logits, 1-labels), axis=-1)   # Average output for false classes
    else: invalid = 0
    return tf.math.reduce_mean(tf.math.multiply(tf.math.pow(1-valid, self.gamma), -tf.math.log(valid + 1e-8))\
                                + self.lamda * invalid, keepdims=True)

class WCE_loss(tf.keras.losses.Loss):
  def __init__(self, loss_kwargs, reduction=tf.keras.losses.Reduction.NONE, name = 'wce_loss'):
    super(WCE_loss, self).__init__(reduction=reduction, name=name)
    self.use_invalids = loss_kwargs["USE_INVALIDS"]
    if self.use_invalids: self.lamda = loss_kwargs["LAMDA"]
    else: self.lamda = 0
    self.num_classes = loss_kwargs["NUM_CLASSES"]
    self.weights = np.array(loss_kwargs["WEIGHTS"]).reshape((-1,1))
    self.weights = np.matmul(np.identity(self.num_classes, dtype=np.float32), self.weights)
    self.weights = tf.convert_to_tensor(self.weights, dtype=tf.float32)

  def call(self, labels, logits):
    """
        loss = (1 - y_hat_for_true_class)^g * -log(y_hat_for_true_class)
    """
    weights = tf.matmul(labels, self.weights) # Weight for each label
    valid = tf.math.reduce_sum(tf.math.multiply(logits, labels), axis=-1, keepdims=True) # Output for true class
    if self.use_invalids: invalid = tf.math.reduce_sum(tf.math.multiply(logits, 1-labels), axis=-1)   # Average output for false classes
    else: invalid = 0
    return tf.math.reduce_mean(tf.math.multiply(weights, -tf.math.log(valid + 1e-8))\
                                + self.lamda * invalid, keepdims=True)

def test_focal_loss(labels, logits):
    loss_kwargs = {"FN": "Focal_loss",
        "GAMMA": 2,
        "USE_INVALIDS": False,
        "LAMDA": 0.01,
        "NUM_CLASSES": 3}
    focal_softmax = Focal_loss(loss_kwargs)(labels, logits).numpy()
    print(focal_softmax.shape)
    print("Focal loss:", focal_softmax, sep='\t')

def test_tversky_loss(labels, logits):
    loss_kwargs = {"FN": "Tversky_loss",
        "ALPHA": 0.7,
        "BETA": 0.3,
        "GAMMA": 0.75}
    tv_loss = Tversky_loss(loss_kwargs)(labels, logits).numpy()
    print(tv_loss.shape)
    print("Tversky loss:", tv_loss, sep='\t')

def test_iou_loss(labels, logits):
    loss_kwargs = {"FN": "Tversky_loss",
        "ALPHA": 1.0,
        "BETA": 1.0,
        "GAMMA": 0.75}
    iou_loss = Tversky_loss(loss_kwargs)(labels, logits).numpy()
    print(iou_loss.shape)
    print("Iou loss:", iou_loss, sep='\t')

def test_wce_loss(labels, logits):
    loss_kwargs = {"FN": "WCE_loss",
        "WEIGHTS": [0.0, 0.6, 0.4],
        "USE_INVALIDS": False,
        "LAMDA": 0.01,
        "NUM_CLASSES": 3}
    wce_loss = WCE_loss(loss_kwargs)(labels, logits).numpy()
    print(wce_loss.shape)
    print("WCE loss:", wce_loss, sep='\t')

def get_loss_fn(loss_fn):
    losses = {
        'Focal_loss': Focal_loss,
        'Tversky_loss': Tversky_loss,
        'WCE_loss': WCE_loss,
        'Lovasz_loss': Lovasz_softmax
    }
    return losses[loss_fn]

def test_lovasz_loss(labels, logits):
    lovasz_softmax = Lovasz_softmax(num_classes=3)(labels, logits).numpy()
    print(lovasz_softmax.shape)
    print("Lovasz loss:", lovasz_softmax, sep='\t')

if __name__ == '__main__':
    labels = tf.constant([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=tf.float32)
    #labels = tf.constant([0, 1, 2, 0], dtype=tf.float32)
    logits = tf.constant([[0.2, 0.2, 0.6], [0.3, 0.5, 0.2], [0.3, 0.5, 0.2], [0.8, 0.1, 0.1]], dtype=tf.float32)
    #logits = tf.constant([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=tf.float32)
    #test_tversky_loss(labels, logits)
    #test_focal_loss(labels, logits)
    #test_iou_loss(labels, logits)
    #tf.print(lovasz_grad(tf.constant([0, 0, 1], dtype=tf.float32)))
    #tf.print(lovasz_softmax_flat(logits, labels, classes='all'))
    test_lovasz_loss(labels, logits)
    #test_wce_loss(labels, logits)
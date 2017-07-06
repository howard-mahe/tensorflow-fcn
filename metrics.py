import tensorflow as tf

# Inspired by https://stackoverflow.com/questions/41617463/tensorflow-confusion-matrix-in-tensorboard
def _createLocalVariable(name, shape, validate_shape=True, dtype=tf.float32):
    """
    Creates a new local variable.
    """
    # Make sure local variables are added to 
    # tf.GraphKeys.LOCAL_VARIABLES
    return tf.contrib.framework.local_variable(
        initial_value=tf.zeros(shape, dtype=dtype),
        name=name,
        validate_shape=validate_shape)

def streamingLoss(loss):
    """
    Compute a streaming loss
    :param loss: loss to accumulate 
    :return: (loss, updateOp, resetOp)
    """    
    # Local variables
    count = _createLocalVariable('streamLossCount', (), dtype=tf.int32)
    acc_loss = _createLocalVariable('streamLossAcc', (), dtype=tf.float32)
    
    # Create the update op for doing a "+=" accumulation on the batch
    countUpdate = count.assign(count + 1)
    lossUpdate = acc_loss.assign(acc_loss + loss)
    updateOp = tf.group(lossUpdate, countUpdate)    
    
    # Create the reset op
    countReset = count.assign(0)    
    lossReset = acc_loss.assign(0.)
    resetOp = tf.group(lossReset, countReset)
    
    # Outputs
    avg_loss = acc_loss / tf.to_float(count)
    
    return avg_loss, updateOp, resetOp
    

def streamingConfusionMatrix(label, prediction, num_classes=None):
    """
    Compute a streaming confusion matrix
    :param label: True labels
    :param prediction: Predicted labels
    :param num_classes: Number of labels for the confusion matrix
    :return: (percentConfusionMatrix, updateOp)
    """
    # Compute a per-batch confusion
    confusion = tf.confusion_matrix(tf.reshape(label, [-1]), tf.reshape(prediction, [-1]),
                                          num_classes=num_classes,
                                          name='acc_confusion')
    
    # Local variables
    count = _createLocalVariable('streamConfusionCount', (), dtype=tf.int32)
    acc_confusion = _createLocalVariable('streamConfusionAcc',
                                     [num_classes, num_classes], dtype=tf.int32)
    
    # Create the update op for doing a "+=" accumulation on the batch
    countUpdate = count.assign(count + tf.reduce_sum(confusion))
    confusionUpdate = acc_confusion.assign(acc_confusion + confusion)
    updateOp = tf.group(confusionUpdate, countUpdate)
    
    # Ouputs the normalized confusion matrix    
    percentConfusion = 100 * tf.truediv(confusion, count)
    
    return percentConfusion, updateOp
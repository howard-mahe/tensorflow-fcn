
from fcn8_vgg import *
from fcn32_vgg import *
from utils import *
from loss import *
from metrics import *
from NYUDv2DataHandler import *

# Configuration variables
OPTIMIZER = 'SGD'
# OPTIMIZER = 'ADAM' 

# Parameters
displayStep  = 50
saveStep     = 10000
testStep     = 1000
assert testStep % displayStep == 0, 'testStep must be a multiple of displayStep for streamingLoss purpose'
test_iter    = 654
max_iter     = 300000

# NYUDv2 dataset
num_classes  = 40
data_type    = 'BGR'
ignore_label = 0

# Shelhamer's 'heavy learning' SGD training strategy for NYUDv2 dataset
batch_size   = 1
sgd_lr       = 1e-10
sgd_momentum = 0.99

# Alternative training strategies
adam_lr = 1e-10

# TODO: implement input data queueing
nyud_data_handler = NYUDv2DataHandler(data_type, num_classes)
sess = tf.InteractiveSession()

# model input and output
with tf.variable_scope('Input'):
    input_image, gt_label = nyud_data_handler.create_placeholders()
    input_keep_probability = tf.placeholder(dtype=tf.float32, name='inputKeepProbability')
    gt_label_squeeze = tf.squeeze(gt_label, axis=3)
    gt_label_onehot = tf.one_hot(gt_label_squeeze, num_classes)

# build model
with tf.name_scope('Model'):
    fcn_vgg = FCN32VGG('../zoo/vgg16/vgg16.npy')
    fcn_vgg.build(input_image, data_type, input_keep_probability, num_classes, random_init_fc8=True)

# image summaries
with tf.name_scope('Evaluations'):
    tf.summary.image('input_image', tf.reverse(input_image, [3]), collections=['train'])
    tf.summary.image('input_image', tf.reverse(input_image, [3]), collections=['test'])
    tf.summary.image('predictions', tf.expand_dims(tf.cast(fcn_vgg.pred_up, tf.uint8), -1), collections=['test']) # display the predictions for the test image at index 'test_iter-1'

# define loss
with tf.name_scope('Loss'):
    weights = tf.cast(gt_label != ignore_label, dtype=tf.float32)
    unnormalized_loss = loss(fcn_vgg.upscore, gt_label_onehot, num_classes, weights)
    batch_loss, update_op_batchloss, reset_op_batchloss = streamingLoss(unnormalized_loss)
    
    # scalar summary
    tf.summary.scalar("loss", batch_loss, collections=['loss'])
    
# define accuracy
# Note: tf.metrics are inherently tricky due to their streaming properties, see https://github.com/tensorflow/tensorflow/issues/9498    
with tf.variable_scope('Metrics'):
    # confusion matrix
    confusion_matrix, update_op_confmat = streamingConfusionMatrix(
        gt_label, fcn_vgg.pred_up, num_classes )
    tf.summary.image('confusion_matrix', 
        tf.reshape(tf.cast(confusion_matrix, tf.float32), [1, num_classes, num_classes, 1]),
        collections=['test'])
    
    # global accuracy
    global_accuracy, update_op_gacc = tf.metrics.accuracy(
        gt_label, fcn_vgg.pred_up )
    tf.summary.scalar('gacc', global_accuracy, collections=['test'])

#    # mean accuracy (per class)
#    mean_accuracy, update_op_macc = tf.metrics.mean_per_class_accuracy(
#        gt_label_squeeze, fcn_vgg.pred_up, num_classes )
#    tf.summary.scalar('macc', mean_accuracy, collections=['test']) # I think it doesn't deal with NaN for unobserved classes
     
#    # mean jaccard index (per class)
#    mean_iu, update_op_miu = tf.metrics.mean_iou(
#        gt_label_squeeze, fcn_vgg.pred_up, num_classes )       
#    tf.summary.scalar('miu', mean_iu, collections=['test']) # I think it doesn't deal with NaN for unobserved classes


# optimizer
with tf.name_scope('Optimizer'):
    if OPTIMIZER == 'SGD':
        optimizer = tf.train.MomentumOptimizer(learning_rate=sgd_lr, momentum=sgd_momentum)
    elif OPTIMIZER == 'ADAM':
        optimizer = tf.train.AdamOptimizer(learning_rate=adam_lr)
    else:
        raise ValueError("OPTIMIZER not in ['SGD', 'ADAM']")
    train = optimizer.minimize(unnormalized_loss)
    
# checkpoint saver
saver = tf.train.Saver()

# Merge all the summaries and write them out to current dir
merged_test = tf.summary.merge([tf.summary.merge_all('loss'),
                                tf.summary.merge_all('test')], 
                               collections='merged_test')
merged_train = tf.summary.merge([tf.summary.merge_all('loss'),
                                 tf.summary.merge_all('train')], 
                                collections='merged_train')
train_writer = tf.summary.FileWriter('./train', sess.graph)
test_writer  = tf.summary.FileWriter('./test')

# Initialize all variables
tf.global_variables_initializer().run() # initialised to VGG16 weights
tf.local_variables_initializer().run()

# Run training loop
print ('=== Training ===')
for i in range(max_iter):
    # get training data
    image_train, label_train = nyud_data_handler.get_sample('train', batch_size)
    
    # Run optimization
    _, _, avg_train_loss, summary = sess.run(
        [train, update_op_batchloss, batch_loss, merged_train],
        feed_dict={input_image: image_train, gt_label: label_train, input_keep_probability: 0.5})
    
    # Display
    if i > 0 and i % displayStep == 0:
        # train loss logs
        train_writer.add_summary(summary, i)
        print ("Iteration: %d, Train Avg (%d) Loss: %.0f" % (i, displayStep, avg_train_loss))
        
        # reset accumulated train loss
        sess.run(tf.local_variables_initializer())

    # Save model weights to disk
    if i > 0 and i % saveStep == 0:
        saver.save(sess, './snapshot/nyud-fcn', global_step=i)
        print ('Model saved: ./snapshot/nyud-fcn')
        
    # Evaluate model on test-test
    if i % testStep == 0:
        # Reset variables
        with tf.variable_scope('test_metrics', reuse=True):
            # reset local variables (total and count)
            tf.local_variables_initializer().run()
            # reset accumulated test loss
            sess.run(tf.local_variables_initializer())

        # Run testing loop
        print ('=== Testing ===')
        for t in range(test_iter):
            # get testing data
            image_test, label_test = nyud_data_handler.get_sample('test', batch_size)
            # run metrics evaluation
            sess.run([update_op_batchloss, update_op_gacc, update_op_confmat],
                feed_dict={input_image: image_test, gt_label: label_test, input_keep_probability: 1.0})

        # test logs
        avg_test_loss, hist, summary = sess.run(
            [batch_loss, confusion_matrix, merged_test],
            feed_dict={input_image: image_test, gt_label: label_test, input_keep_probability: 1.0})
        test_writer.add_summary(summary, i)

        # Compute my metrics
        np.seterr(divide='ignore', invalid='ignore')        
        acc = np.diag(hist) / hist.sum(1)
        mean_acc = np.nanmean(acc)
        iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
                                    
        # Logs my metrics
        summary = tf.Summary(value=[
            tf.Summary.Value(tag="Metrics/my_macc", simple_value=mean_acc),
            tf.Summary.Value(tag="Metrics/my_miu", simple_value=mean_iu)                                    
            ])
        test_writer.add_summary(summary, i)                                
        print ("Test Acc (%d) - loss: %.0f, my_macc: %.2f%%, my_mIoU: %.2f%%" % (test_iter, avg_test_loss, mean_acc*100, mean_iu*100))

        # Resume training loop
        print ('=== Training ===')
        sess.run(tf.local_variables_initializer())
        

train_writer.close()
test_writer.close()


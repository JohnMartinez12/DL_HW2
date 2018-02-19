
from __future__ import print_function

import tensorflow as tf
import numpy as np 
import scipy.misc
import glob 
import os
import sys 

def get_img_array(path):
  """
  Given path of image, returns it's numpy array
  """
  return scipy.misc.imread(path)

def get_files(folder):
  """
  Given path to folder, returns list of files in it
  """
  filenames = [file for file in glob.glob(folder+'*/*')]
  filenames.sort()
  return filenames

def get_label(filepath, label2id):
  """
  Files are assumed to be labeled as: /path/to/file/999_frog.png
  Returns label for a filepath
  """
  tokens = filepath.split('/')
  label = tokens[-1].split('_')[1][:-4]
  if label in label2id:
    return label2id[label]
  else:
    sys.exit("Invalid label: " + label)
    
# Functions to load data, DO NOT change these

def get_labels(folder, label2id):
  """
  Returns vector of labels extracted from filenames of all files in folder
  :param folder: path to data folder
  :param label2id: mapping of text labels to numeric ids. (Eg: automobile -> 0)
  """
  files = get_files(folder)
  y = []
  for f in files:
    y.append(get_label(f,label2id))
  return np.array(y)

def one_hot(y, num_classes=10):
  """
  Converts each label index in y to vector with one_hot encoding
  One-hot encoding converts categorical labels to binary values
  """
  y_one_hot = np.zeros((num_classes, y.shape[0]))
  y_one_hot[y, range(y.shape[0])] = 1
  return y_one_hot

def get_label_mapping(label_file):
  """
  Returns mappings of label to index and index to label
  The input file has list of labels, each on a separate line.
  """
  print(os.listdir())
  with open(label_file, 'r') as f:
    id2label = f.readlines()
    id2label = [l.strip() for l in id2label]
  label2id = {}
  count = 0
  for label in id2label:
    label2id[label] = count
    count += 1
  return id2label, label2id

def get_images(folder):
  """
  returns numpy array of all samples in folder
  each column is a sample resized to 30x30 and flattened
  """
  files = get_files(folder)
  images = []
  count = 0
  
  for f in files:
    count += 1 
    if count % 10000 == 0:
      print("Loaded {}/{}".format(count,len(files)))
    img_arr = get_img_array(f)
    img_arr = img_arr.flatten() / 255.0
    images.append(img_arr)
  X = np.column_stack(images)

  return X

def get_train_data(data_root_path):
  """
  Return X and y
  """
  train_data_path = data_root_path + 'train'
  id2label, label2id = get_label_mapping(data_root_path+'labels.txt')
  print(label2id)
  X = get_images(train_data_path)
  y = get_labels(train_data_path, label2id)
  return X, y

def save_predictions(filename, y):
  """
  Dumps y into .npy file
  """
  np.save(filename, y)
  
# Load the data
data_root_path = 'HW2_data/'
X_train, y_train = get_train_data(data_root_path) # this may take a few minutes
X_test = get_images(data_root_path + 'test')


def get_batch(X, y, batch_size):
  """
  Return minibatch of samples and labels
  
  :param X, y: samples and corresponding labels
  :parma batch_size: minibatch size
  :returns: (tuple) X_batch, y_batch
  """
  # Random indices for the samples
  indices = np.random.randint(y.shape[0]-1, size= batch_size)
  
  X_batch = X[:, indices]
  y_batch = y[indices]
  
  return X_batch, y_batch

#==========================================================================================

# Parameters
learning_rate = 0.001
training_iters = 200000
batch_size = 2500
display_step = 10

# Network Parameters
n_input = 3072 
n_classes = 10 
dropout = 0.75 

# tf Graph input
x = tf.placeholder(tf.float32, [n_input, None])
y = tf.placeholder(tf.float32, [n_classes, None])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)


# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 32, 32, 3])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 3, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([8*8*64, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=tf.transpose(y)))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 0))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    
    while step * batch_size < training_iters:
        batch_x, batch_y = get_batch(X_train, y_train, batch_size)
        #batch_x = batch_x.T
        batch_y = one_hot(batch_y)

        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                       keep_prob: dropout})
        if step % display_step == 0:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                              y: batch_y,
                                                              keep_prob: 1.})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")

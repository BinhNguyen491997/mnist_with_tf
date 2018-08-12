import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

img_size = 28
img_size_flat = img_size * img_size

n_classes = 10
x = tf.placeholder('float', [None, img_size_flat])
y = tf.placeholder('float', [None, n_classes])

def conv2d(data, weights):
    return tf.nn.conv2d(data, weights, strides=[1, 1, 1, 1], padding='SAME')

def maxpool2d(data):
    return tf.nn.max_pool(data, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def convolutional_neural_network(data):
    weights = {
        'w_conv1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
        'w_conv2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
        'w_fc': tf.Variable(tf.random_normal([7*7*64, 1024])),
        'out': tf.Variable(tf.random_normal([1024, n_classes])),
    }
    biases = {
        'b_conv1': tf.Variable(tf.random_normal([32])),
        'b_conv2': tf.Variable(tf.random_normal([64])),
        'b_fc': tf.Variable(tf.random_normal([1024])),
        'out': tf.Variable(tf.random_normal([n_classes])),
    }

    data = tf.reshape(data, shape=[-1, img_size, img_size, 1])
    conv1 = conv2d(data, weights['w_conv1'])
    conv1 = maxpool2d(conv1)
    conv2 = conv2d(conv1, weights['w_conv2'])
    conv2 = maxpool2d(conv2)

    fc = tf.reshape(conv2, [-1, 7*7*64])
    fc = tf.nn.relu(tf.matmul(fc, weights['w_fc']) + biases['b_fc'])
    output = tf.matmul(fc, weights['out']) + biases['out']
    return output

def train_neural_network(data):
    prediction = convolutional_neural_network(data)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y)
    cost = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    batch_size = 100
    hm_epochs = 10
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples / batch_size)):
                batch_x, batch_y = mnist.train.next_batch(batch_size)
                
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                y: batch_y})

                epoch_loss += c
            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:',epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        # Print the accuracy
        print('Accuracy:', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

train_neural_network(x)
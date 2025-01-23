import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

def load_dataset():
    df = pd.read_csv('pulsar_stars.csv')

    features = df[[\
                    'Mean of the integrated profile',\
                    'Standard deviation of the integrated profile',\
                    'Excess kurtosis of the integrated profile',\
                    'Skewness of the integrated profile',\
                    'Mean of the DM-SNR curve',\
                    'Standard deviation of the DM-SNR curve',\
                    'Excess kurtosis of the DM-SNR curve',\
                    'Skewness of the DM-SNR curve']]
    
    target = df[['target_class']]

    encoder = OneHotEncoder(sparse = False)
    target = encoder.fit_transform(target)

    return features, target

features, target = load_dataset()

scaler = MinMaxScaler()
features = scaler.fit_transform(features)

train_features, test_features, train_target, test_target = train_test_split(features, target, test_size = 0.2)

layer = {
    'input': 8,
    'hidden': 10,
    'output': 2, 
}

weight = {
    'input-hidden': tf.Variable(tf.random_normal([layer['input'], layer['hidden']])),
    'hidden-output': tf.Variable(tf.random_normal([layer['hidden'], layer['output']])),
}

bias = {
    'input-hidden': tf.Variable(tf.random_normal([layer['hidden']])),
    'hidden-output': tf.Variable(tf.random_normal([layer['output']])),
}

features_placeholder = tf.placeholder(tf.float32, [None, layer['input']])
target_placeholder = tf.placeholder(tf.float32, [None, layer['output']])

def feed_forward():
    y1 = tf.matmul(features_placeholder, weight['input-hidden']) + bias['input-hidden']
    y1 = tf.nn.sigmoid(y1)

    y2 = tf.matmul(y1, weight['hidden-output']) + bias['hidden-output']
    return tf.nn.sigmoid(y2)

prediction = feed_forward()
error = tf.reduce_mean(.5 * (target_placeholder - prediction) ** 2)

learningrate = .1

optimizer = tf.train.GradientDescentOptimizer(learningrate)
train = optimizer.minimize(error)

epoch = 5000

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    feed_train = {
        features_placeholder: train_features,
        target_placeholder: train_target
    }

    for i in range(1, epoch + 1):
        sess.run(train, feed_dict=feed_train)

        if i % 200 == 0:
            result_loss = sess.run(error, feed_dict=feed_train)
            print("epoch: {}, error: {}".format(i, result_loss))
    
    # accuracy
    feed_test = {
        features_placeholder: test_features,
        target_placeholder: test_target
    }

    match = tf.equal(tf.argmax(target_placeholder, axis=1), tf.argmax(prediction, axis=1))
    cast = tf.cast(match, tf.float32)
    accuracy = tf.reduce_mean(cast)

    print("Accuracy: {}".format(sess.run(accuracy, feed_dict=feed_test)))
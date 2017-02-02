import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image

im = Image.open("input.png")
im_arr = list(im.getdata())
im_nparr = [[]]

for x in im_arr:
    x = (1 - x[0]/255, 1 - x[1]/255, 1 - x[2]/255)
    im_nparr = np.append(im_nparr, [[(x[0] + x[1] + x[2])/3]], axis=1)

print(im_nparr.shape)

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

n_nodes_hl1 = 784
n_nodes_hl2 = 784
n_nodes_hl3 = 784

n_classes = 10
batch_size = 1000

x = tf.placeholder("float", [None, 784])
y = tf.placeholder("float")

reutilizar_valores = True
hm_epochs = 0

def train_neural_network(x):

    data = x
    
    if reutilizar_valores:
        
        hidden_1_layer = {"weights":tf.Variable(np.loadtxt("savew_hl1", dtype="float32")),
                          "biases":tf.Variable(np.loadtxt("saveb_hl1", dtype="float32"))}

        hidden_2_layer = {"weights":tf.Variable(np.loadtxt("savew_hl2", dtype="float32")),
                          "biases":tf.Variable(np.loadtxt("saveb_hl2", dtype="float32"))}

        hidden_3_layer = {"weights":tf.Variable(np.loadtxt("savew_hl3", dtype="float32")),
                          "biases":tf.Variable(np.loadtxt("saveb_hl3", dtype="float32"))}

        output_layer = {"weights":tf.Variable(np.loadtxt("savew_op", dtype="float32")),
                          "biases":tf.Variable(np.loadtxt("saveb_op", dtype="float32"))}
        
    else:
        
        hidden_1_layer = {"weights":tf.Variable(tf.random_normal([784, n_nodes_hl1])),
                          "biases":tf.Variable(tf.random_normal([n_nodes_hl1]))}

        hidden_2_layer = {"weights":tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                          "biases":tf.Variable(tf.random_normal([n_nodes_hl2]))}

        hidden_3_layer = {"weights":tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                          "biases":tf.Variable(tf.random_normal([n_nodes_hl3]))}

        output_layer = {"weights":tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                          "biases":tf.Variable(tf.random_normal([n_classes]))}

    l1 = tf.add(tf.matmul(data, hidden_1_layer["weights"]), hidden_1_layer["biases"])
    l1 = tf.nn.relu(l1)
        
    l2 = tf.add(tf.matmul(l1, hidden_2_layer["weights"]), hidden_2_layer["biases"])
    l2 = tf.nn.relu(l2)
        
    l3 = tf.add(tf.matmul(l2, hidden_3_layer["weights"]), hidden_3_layer["biases"])
    l3 = tf.nn.relu(l3)

    prediction = tf.matmul(l3, output_layer["weights"]) + output_layer["biases"]
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict = {x: epoch_x, y: epoch_y})
                epoch_loss += c
            print("Epoch", epoch, "completed out of", hm_epochs, "loss:", epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, "float"))
        print("Accuracy:", accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

        savew_hl1 = hidden_1_layer["weights"].eval()
        saveb_hl1 = hidden_1_layer["biases"].eval()
        savew_hl2 = hidden_2_layer["weights"].eval()
        saveb_hl2 = hidden_2_layer["biases"].eval()
        savew_hl3 = hidden_3_layer["weights"].eval()
        saveb_hl3 = hidden_3_layer["biases"].eval()
        savew_op = output_layer["weights"].eval()
        saveb_op = output_layer["biases"].eval()

        resultado = []

        for i in range(10):
            tmp = [[0,0,0,0,0,0,0,0,0,0]]
            tmp[0][i] = 1
            resultado = np.append(resultado, accuracy.eval({x:im_nparr, y:tmp}))

        print(array_para_numero_resultado(resultado))

    np.savetxt("savew_hl1", savew_hl1)
    np.savetxt("saveb_hl1", saveb_hl1)
    np.savetxt("savew_hl2", savew_hl2)
    np.savetxt("saveb_hl2", saveb_hl2)
    np.savetxt("savew_hl3", savew_hl3)
    np.savetxt("saveb_hl3", saveb_hl3)
    np.savetxt("savew_op", savew_op)
    np.savetxt("saveb_op", saveb_op)

def array_para_numero_resultado(res):
    for i in range(len(res)):
        if res[i] == 1:
            return int(i)

train_neural_network(x)

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

df =input_data.read_data_sets("/tmp/data/",one_hot=True)

img_dim=28*28
gen_dim=256
disc_dim=256
noise_dim=100


batch_size=128
num_steps=100000
learning_rate=2e-4

def weight_init(shape):
    return tf.random_normal(shape=shape,stddev=1./tf.sqrt(shape[0]/2))

W={"w1":tf.Variable(weight_init([noise_dim,gen_dim])),
   "w2": tf.Variable(weight_init([gen_dim, img_dim])),
   "w3": tf.Variable(weight_init([img_dim, disc_dim])),
   "w4": tf.Variable(weight_init([disc_dim, 1]))}

b={"b1":tf.Variable(tf.zeros([gen_dim])),
   "b2": tf.Variable(tf.zeros([img_dim])),
   "b3": tf.Variable(tf.zeros([disc_dim])),
   "b4": tf.Variable(tf.zeros([1]))}

def generator(x):
    ret=tf.matmul(x,W["w1"])
    ret=tf.add(ret,b["b1"])
    ret=tf.nn.relu(ret)

    ret = tf.matmul(ret, W["w2"])
    ret = tf.add(ret, b["b2"])
    ret = tf.nn.sigmoid(ret)

    return ret


def discriminator(x):
    ret = tf.matmul(x, W["w3"])
    ret = tf.add(ret, b["b3"])
    ret = tf.nn.relu(ret)

    ret = tf.matmul(ret, W["w4"])
    ret = tf.add(ret, b["b4"])
    ret = tf.nn.sigmoid(ret)

    return ret


gen_inp=tf.placeholder(tf.float32,shape=[None,noise_dim])
disc_inp=tf.placeholder(tf.float32,shape=[None,img_dim])

gen_out=generator(gen_inp)
disc_real_out=discriminator(disc_inp)
disc_fake_out=discriminator(gen_out)

optimize_gen=tf.train.AdamOptimizer(learning_rate=learning_rate)
optimize_disc=tf.train.AdamOptimizer(learning_rate=learning_rate)

cost_gen=-tf.reduce_mean(tf.log(disc_fake_out))
cost_disc=-tf.reduce_mean(tf.log(disc_real_out)+tf.log(1.-disc_fake_out))

var_gen=[W["w1"],W["w2"],b["b1"],b["b2"]]
var_disc=[W["w3"],W["w4"],b["b3"],b["b4"]]

train_gen=optimize_gen.minimize(cost_gen,var_list=var_gen)
train_disc=optimize_disc.minimize(cost_disc,var_list=var_disc)

init=tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for step in range(1,num_steps):
        batch_x,_=df.train.next_batch(batch_size)
        noise_temp=np.random.uniform(-1.,1.,size=[batch_size,noise_dim])
        feed_dict={disc_inp:batch_x,gen_inp:noise_temp}
        _,_,gl,dl=sess.run([train_gen,train_disc,cost_gen,cost_disc],feed_dict=feed_dict)
        if step%2000==0:
            print("step: %i generator loss: %f,discriminator loss: %f"%(step,gl,dl))

    test_size=5
    canvas=np.empty((28*test_size,28*test_size))

    for i in range(test_size):
        z=np.random.uniform(-1.,1.,size=[test_size,noise_dim])
        g=sess.run(gen_out,feed_dict={gen_inp:z})
        g=-1*(g-1)
        for j in range(test_size):
            canvas[i*28:(i+1)*28,j*28:(j+1)*28]=g[j].reshape([28,28])
plt.figure(figsize=(test_size,test_size))
plt.imshow(canvas,origin="upper",cmap="gray")
plt.show()

# def run():
#     print("1. Train")
#     print("2. Test)

# if __name__=="__main__":
#     run()

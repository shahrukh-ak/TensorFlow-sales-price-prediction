## linear regression
## housing pricing based on square footage and price

import tensorflow as tf
import numpy as np
#########################

def inference(x):
    W=tf.variable(tf.zeros([1,1])
    b=tf.variable(tf.zeros([1]))                          #we only have X1 so only w1
    y=tf.matmul(W, x) + b
    return y

############################


def loss(y,y_):
    cost=tf.reduce_sum(tf.pow( (y_ - y), 2 ))
    return cost


#############################

def training():
    train_step=tf.train.GradeintDescentOptimizer(0.00001).minimize(cost)  ## minimize cost
    return train_step

###############################


def evaluate(y,y_):
    correct_prediction=y
    float_val=tf.cast(correct_prediction, tf.float32)   ###3converting to float
    return float_val

#################################
##y=w1x1+b
x=tf.placeholder(tf.float32, [None,1])     ##only square footage
y_=tf.placeholder(tf.float32, [None,1])    #actual
y= inference(x)      # predicted one with x =data
cost=loss(y, y_)   #compare real and pedicted

train_step=training(cost)
eval_op=evaluate(y,y_)

#####################################


init=tf.initialize_all_variables()
sess=tf.session()
session.run(init)  ### all variables get initialized


######################################33333
###training model
steps=100
for i in range(steps):
    xs=np.array([[i]])   ## house size
    ys=np.array([[5*i]])  ## house price 5 times house size
    feed ={x:xs, y_:ys}
    sess.run(train_step, feed_dict=feed)


##################################3
### testing model

for i in range(100,200):
    xs_test=np.array([[i]])  ## house price
    ys_test=np.array([[2*i]])  ## if testing correctly it should predict 500 - 1000
    feed_test={x:xs_test, y_:ys_test}
    result=sess.run(eval_op, feed_dict = feed_test)
    print "Run {},{}".format(i, result)
    r=raw_input

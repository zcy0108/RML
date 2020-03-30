from __future__ import print_function
import tensorflow as tf
from tensorflow.python.framework import ops
import os


def main():
    # The default path for saving event files is the same folder of this python file.
    tf.app.flags.DEFINE_string('log', os.path.dirname(os.path.abspath(__file__)) + '/logs',
                               'Directory where event logs are written to.')
    # Store all elements in FLAG structure!
    FLAGS = tf.app.flags.FLAGS
    if not os.path.isabs(os.path.expanduser(FLAGS.log)):
        raise ValueError('You must assign absolute path for --log')
    # ------------------ upper code is for log ------------------

    # Create three variables with some default values.
    weights = tf.Variable(tf.random.normal([2, 3], stddev=0.1),
                          name="weights")
    biases = tf.Variable(tf.zeros([3]), name="biases")
    custom_variable = tf.Variable(tf.zeros([3]), name="custom")

    # Get all the variables' tensors and store them in a list.
    all_variables_list = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)

    # "variable_list_custom" is the list of variables that we want to initialize.
    variable_list_custom = [weights, custom_variable]

    # The initializer
    init_custom_op = tf.variables_initializer(var_list=variable_list_custom)
    # Method-1
    # Add an op to initialize the variables.
    init_all_op = tf.global_variables_initializer()

    # Method-2
    init_all_op = tf.variables_initializer(var_list=all_variables_list)
    # Create another variable with the same value as 'weights'.
    WeightsNew = tf.Variable(weights.initialized_value(), name="WeightsNew")

    # Now, the variable must be initialized.
    init_WeightsNew_op = tf.variables_initializer(var_list=[WeightsNew])
    with tf.Session() as sess:
        # Run the initializer operation.
        sess.run(init_all_op)
        sess.run(init_custom_op)
        sess.run(init_WeightsNew_op)
        writer = tf.summary.FileWriter(os.path.expanduser(FLAGS.log), sess.graph)

    writer.close()

    return


if __name__ == "__main__":
    main()

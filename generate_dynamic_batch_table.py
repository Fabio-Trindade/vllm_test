import tensorflow as tf

import tensorflow as tf

# Caminho do log do TensorBoard
log_dir = "logs/"

# Carrega os eventos do TensorFlow
events = tf.compat.v1.train.summary_iterator(log_dir)
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU') 
print("GPUs:",  physical_devices)

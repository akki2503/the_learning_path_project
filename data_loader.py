import tensorflow as tf
import os
class get_record_dataset(object):
    def __init__(self, filenames, batch_size, num_classes):
        self.filenames = filenames
        self.batch_size = batch_size
        self.num_classes = num_classes

    def decode_image(self, image):
        self.image = tf.image.decode_jpeg(image, channels=3)
        self.image = tf.cast(self.image, tf.float32)
        self.image = tf.image.resize(self.image, [224,224])
        self.image = tf.image.random_flip_left_right(self.image) #augmentations
        return self.image

    def read_tfrecord(self, example):
        tfrecord_format = ({
            'image': tf.io.FixedLenFeature([], tf.string),
            'target': tf.io.FixedLenFeature([], tf.int64),
        })                                                  #defining tfrecord format
        example = tf.io.parse_single_example(example, tfrecord_format)
        image = self.decode_image(example['image'])
        label = tf.one_hot(example['target'], self.num_classes, on_value=1, off_value=0)
        return image, label
  
    def load_dataset(self, filenames):
        self.ignore_order = tf.data.Options()
        self.ignore_order.experimental_deterministic = False
        self.dataset = tf.data.TFRecordDataset(self.filenames)
        self.dataset = self.dataset.with_options(self.ignore_order)
        self.dataset = self.dataset.map(self.read_tfrecord, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return self.dataset

    def get_dataset(self):
        self.dataset = self.load_dataset(self.filenames)
        self.dataset = self.dataset.shuffle(len(self.dataset))
        self.dataset = self.dataset.batch(self.batch_size)
        self.dataset = self.dataset.prefetch(buffer_size  = tf.data.experimental.AUTOTUNE)
        return self.dataset



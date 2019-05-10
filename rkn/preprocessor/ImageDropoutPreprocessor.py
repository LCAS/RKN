import tensorflow as tf

class ImageDropoutPreprocessor:


    def __init__(self, min_image_dropout, max_image_droput, prefix_length=5):

        self.min_image_dropout = min_image_dropout
        self.max_image_dropout = max_image_droput
        self.prefix_length = prefix_length

    def __call__(self,  observations, on_gpu):
        batch_size = tf.shape(observations)[0]
        seq_length = tf.shape(observations)[1]

        self.factors = self._sample_factors(batch_size, seq_length)

        expanded_factors = tf.expand_dims(tf.expand_dims(tf.expand_dims(self.factors, -1), -1), -1)
        self.preprocessed_obs = expanded_factors * observations
        return self.preprocessed_obs


    def _sample_factors(self, batch_size, seq_length):

        max_samples_needed_f = tf.ceil((seq_length - self.prefix_length) / self.min_image_dropout)
        max_samples_needed = tf.to_int32(max_samples_needed_f)
        samples = tf.random_uniform(shape=[batch_size, max_samples_needed],
                                    minval=self.min_image_dropout,
                                    maxval=self.max_image_dropout + 1, # maxval iteself is excluded
                                    dtype=tf.int32)
        keep_image_idx = tf.cumsum(samples, -1)

        factor_array = tf.TensorArray(dtype=tf.float32, size=seq_length - self.prefix_length)
        loop_init = (tf.constant(0), factor_array)
        loop_condiction = lambda i, _ : tf.less(i, seq_length - self.prefix_length)

        def loop_body(i, arr):
            arr = arr.write(i, tf.cast(tf.reduce_any(tf.equal(i, keep_image_idx), -1), tf.float32))
            return i + 1, arr

        loop_final = tf.while_loop(loop_condiction, loop_body, loop_init, parallel_iterations=1) # this should be more..
        factors_transposed = loop_final[1].stack()
        return tf.concat([tf.ones([batch_size, self.prefix_length]), tf.transpose(factors_transposed)], -1)


if __name__ == '__main__':

    idp = ImageDropoutPreprocessor(2, 5, 5)
    obs = tf.ones([5, 150, 8, 8, 3])
    sess = tf.InteractiveSession()
    dout_obs_tensor = idp(obs, False)

    factors, dout_obs_tensor = sess.run(fetches=(idp.factors, dout_obs_tensor))
    print(factors)


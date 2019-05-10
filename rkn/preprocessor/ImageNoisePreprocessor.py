import tensorflow as tf

class ImageNoisePreprocessor:
    """Can be given as a preprocessor to the rkn model, adds temporarily correlated noise to the images
     (as described in paper)"""

    def __init__(self, r=0.2, t_ll=0.1, t_lu=0.4, t_ul=0.6, t_uu=0.9):
        """
        :param r: Interval bounds (lower bound = -r) to sample the additional noise from
        :param t_ll: Lower bound of the interval the lower threshold is sampled from
        :param t_lu: Upper bound of the interval the lower threshold is sampled from
        :param t_ul: lower bound of the interval the upper threshold is sampled from
        :param t_uu: upper bound of the interval the upper threshold is sampled from
        """
        assert t_ll <= t_lu <= t_ul <= t_uu, "Invalid bounds for noise generation"

        self.r = r
        self.t_ll = t_ll
        self.t_lu = t_lu
        self.t_ul = t_ul
        self.t_uu = t_uu
        self.prefix_length = 5

    def __call__(self, observations, on_gpu):
        """
        Adds nodes for sampling to graph
        :param observations: tensor holding observations
        :param on_gpu: indicating the image format, not relevant for this preprocessor
        :return: tensor with noisy observation
        """
        random_img = tf.random_uniform(shape=tf.shape(observations), minval=0.0, maxval=1.0)

        batch_size = tf.shape(observations)[0]
        seq_length = tf.shape(observations)[1]

        self.t1 = tf.random_uniform(shape=[batch_size, 1], minval=self.t_ll, maxval=self.t_lu)
        self.t2 = tf.random_uniform(shape=[batch_size, 1], minval=self.t_ul, maxval=self.t_uu)

        self.sampled_factors = self.sample_correlated_factors(batch_size, seq_length)
        factors = (self.sampled_factors - self.t1) / (self.t2 - self.t1)
        self.factors = tf.maximum(0.0, tf.minimum(factors, 1.0))

        expanded_factors = tf.expand_dims(tf.expand_dims(tf.expand_dims(self.factors, axis=-1), axis=-1), axis=-1)
        self.preprocessed_obs = expanded_factors * observations + (1 - expanded_factors) * random_img

        return self.preprocessed_obs

    def sample_correlated_factors(self, batch_size, seq_length):
        """ Samples the correlated factors
        :param batch_size:
        :param seq_length:
        :return:
        """
        #Internally we need to work with [seq_length x batch_size], contrary to everywhere else
        rand_array = tf.TensorArray(dtype=tf.float32, size=seq_length - self.prefix_length, clear_after_read=False)
        rand_array = rand_array.write(0, tf.random_uniform(shape=[1, batch_size],
                                                           minval=0, maxval=1))

        loop_init = (tf.constant(0), rand_array)
        loop_condition = lambda i, rand_array: tf.less(i, seq_length - (self.prefix_length + 1))
        def loop_body(i, rand_array):

            old_rand = rand_array.read(i)
            new_rand = old_rand + tf.random_uniform(shape=[1, batch_size], minval=-self.r, maxval=+self.r)
            new_rand = tf.maximum(0.0, tf.minimum(new_rand, 1.0))
            rand_array = rand_array.write(i + 1, new_rand)
            return i + 1, rand_array

        loop_final = tf.while_loop(loop_condition, loop_body, loop_init, parallel_iterations=1)
        transposed_factors = loop_final[1].concat()
        return tf.concat([tf.ones([batch_size, self.prefix_length]), tf.transpose(transposed_factors)], 1)
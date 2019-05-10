import tensorflow as tf
import numpy as np


def diag_a_diag_at(A, diag_mat):
    """Batched computation of diagonal entries of (A * diag_mat * A^T) where A is a square matrix and diag_mat is a
    batch of diagonal matrices (represented as vectors containing diagonal entries)
    :param A: square matrix,
    :param diag_mat: batch of diagonal matrices (represented as vecotrs containing diagonal entries
    :returns diagonal entries of  A * diag_mat * A^T"""
    diag_ext = tf.expand_dims(diag_mat, 1)
    A = tf.expand_dims(A, 0) if len(A.get_shape()) == 2 else A
    first_prod = tf.square(A) * diag_ext
    return tf.reduce_sum(first_prod, axis=2)

def diag_a_diag_bt(A, diag_mat, B):
    """Batched computation of diagonal entries of (A * diag_mat * B^T) where A and B are square matrices and diag_mat is a
    batch of diagonal matrices (represented as vectors containing diagonal entries)
    :param A: square matrix,
    :param diag_mat: batch of diagonal matrices (represented as vecotrs containing diagonal entries
    :returns diagonal entries of  A * diag_mat * A^T"""
    diag_ext = tf.expand_dims(diag_mat, 1)
    A = tf.expand_dims(A, 0) if len(A.get_shape()) == 2 else A
    B = tf.expand_dims(B, 0) if len(B.get_shape()) == 2 else B
    first_prod = A * B * diag_ext
    return tf.reduce_sum(first_prod, axis=2)

def diag_a_diag_at_batched(A, diag_mat):
    """Batched computation of diagonal entries of (A * diag_mat * A^T) where A is a batch of square matrices and diag_mat is a
    batch of diagonal matrices (represented as vectors containing diagonal entries)
    :param A: square matrix,
    :param diag_mat: batch of diagonal matrices (represented as vecotrs containing diagonal entries
    :returns diagonal entries of  A * diag_mat * A^T"""
    diag_ext = tf.expand_dims(diag_mat, 1)
    first_prod = tf.square(A) * diag_ext
    return tf.reduce_sum(first_prod, axis=2)

def diag_a_diag_bt_batched(A, diag_mat, B):
    """Batched computation of diagonal entries of (A * diag_mat * B^T) where A and B are batches of square matrices and diag_mat is a
    batch of diagonal matrices (represented as vectors containing diagonal entries)
    :param A: square matrix,
    :param diag_mat: batch of diagonal matrices (represented as vecotrs containing diagonal entries
    :returns diagonal entries of  A * diag_mat * A^T"""
    diag_ext = tf.expand_dims(diag_mat, 1)
    first_prod = A * B * diag_ext
    return tf.reduce_sum(first_prod, axis=2)

def bmatvec(mat, vec):
    return tf.squeeze(tf.matmul(mat, tf.expand_dims(vec, axis=-1)), axis=-1)

# Works like np.matmul(a, b) for a 2 and a 3 dim matrix
def bmatmul(a, b, transpose_a=False, transpose_b=False):
    with tf.name_scope('BMatmul'):
        a_shape = a.get_shape()
        tf.layers.dense
        b_shape = b.get_shape()
        assert len(a_shape) == 2 and len(b_shape) == 3 or len(a_shape) == 3 and len(b_shape) == 2,\
            'bmatmul umatching lenghts'
        a_axis1 = 1 if transpose_a else 0
        a_axis2 = 0 if transpose_a else 1
        b_axis1 = 1 if transpose_b else 0
        b_axis2 = 0 if transpose_b else 1

        if len(a.get_shape()) == 3:
            res = tf.tensordot(a, b, axes=[[a_axis2 + 1], [b_axis1]])
            res.set_shape([a_shape[0], a_shape[a_axis1 + 1], b_shape[b_axis2]])
        else:
            res = tf.transpose(tf.tensordot(a, b, axes=[[a_axis2], [b_axis1 + 1]]), perm=[1, 0, 2])
            res.set_shape([b_shape[0], a_shape[a_axis1], b_shape[b_axis2 + 1]])
        return res



"""Testing"""
def numpy_ref_diag_a_diag_at(a, diag_mat):
    batch_size = diag_mat.shape[0]
    dim = diag_mat.shape[1]
    np_diag_wrapper = np.zeros([batch_size, dim, dim])
    for i in range(batch_size):
        np_diag_wrapper[i, :, :] = np.diag(diag_mat[i, :])

    res = np.matmul(np.matmul(a, np_diag_wrapper), a.transpose())
    return np.diagonal(res, axis1=1, axis2=2)

def numpy_ref_diag_a_diag_bt(a, diag_mat, b):
    batch_size = diag_mat.shape[0]
    dim = diag_mat.shape[1]
    np_diag_wrapper = np.zeros([batch_size, dim, dim])
    for i in range(batch_size):
        np_diag_wrapper[i, :, :] = np.diag(diag_mat[i, :])

    res = np.matmul(np.matmul(a, np_diag_wrapper), b.transpose())
    return np.diagonal(res, axis1=1, axis2=2)

def numpy_ref_bmatmul(a, b, transposeA=False, transposeB=False):
    if len(a.shape) == 2:
        a = np.expand_dims(a, 0)
    elif len(b.shape) == 2:
        b = np.expand_dims(b, 0)
    if transposeA:
        a = np.transpose(a, axes=[0, 2, 1])
    if transposeB:
        b = np.transpose(b, axes=[0, 2, 1])
    return np.matmul(a, b)

def test_diag_a_diag_at():

    dim = 100
    batch_size = 20

    a_ph = tf.placeholder(tf.float64, shape=[dim, dim])
    diag_ph = tf.placeholder(tf.float64, shape=[batch_size, dim])

    res = diag_a_diag_at(a_ph, diag_ph)

    sess = tf.InteractiveSession()

    print("Test Diag A Diag AT")
    for i in range(10):

        a = np.random.uniform(low=-2.0, high=2.0, size=[dim, dim])
        diag = np.random.uniform(low=-2.0, high=2.0, size=[batch_size, dim])

        np_res = numpy_ref_diag_a_diag_at(a, diag)
        tf_res = sess.run(fetches=res, feed_dict={a_ph: a, diag_ph: diag})
        assert np.all(np.isclose(np_res, tf_res)), "Test Diag a diag a^T failed"


def test_diag_a_diag_bt():

    dim = 100
    batch_size = 20

    a_ph = tf.placeholder(tf.float64, shape=[dim, dim])
    b_ph = tf.placeholder(tf.float64, shape=[dim, dim])
    diag_ph = tf.placeholder(tf.float64, shape=[batch_size, dim])

    res = diag_a_diag_bt(a_ph, diag_ph, b_ph)

    sess = tf.InteractiveSession()

    print("Test Diag A Diag BT")
    for i in range(10):

        a = np.random.uniform(low=-2.0, high=2.0, size=[dim, dim])
        b = np.random.uniform(low=-2.0, high=2.0, size=[dim, dim])
        diag = np.random.uniform(low=-2.0, high=2.0, size=[batch_size, dim])

        np_res = numpy_ref_diag_a_diag_bt(a, diag, b)
        tf_res = sess.run(fetches=res, feed_dict={a_ph: a, diag_ph: diag, b_ph: b})
        assert np.all(np.isclose(np_res, tf_res)), "Test Diag a diag b^T failed"

def test_bmatmul():
    dim1 = 100
    dim2 = 50
    batch_size = 20

    a2 = tf.placeholder(tf.float64, shape=[dim1, dim2])
    a2T = tf.placeholder(tf.float64, shape=[dim2, dim1])
    a3 = tf.placeholder(tf.float64, shape=[batch_size, dim1, dim2])
    a3T = tf.placeholder(tf.float64, shape=[batch_size, dim2, dim1])
    b2 = tf.placeholder(tf.float64, shape=[dim2, dim1])
    b2T = tf.placeholder(tf.float64, shape=[dim1, dim2])
    b3 = tf.placeholder(tf.float64, shape=[batch_size, dim2, dim1])
    b3T = tf.placeholder(tf.float64, shape=[batch_size, dim1, dim2])

    ta2b3 = bmatmul(a2, b3, transpose_a=False, transpose_b=False)
    ta2tb3 = bmatmul(a2T, b3, transpose_a=True, transpose_b=False)
    ta2b3t = bmatmul(a2, b3T, transpose_a=False, transpose_b=True)
    ta2tb3t = bmatmul(a2T, b3T, transpose_a=True, transpose_b=True)

    ta3b2 = bmatmul(a3, b2, transpose_a=False, transpose_b=False)
    ta3tb2 = bmatmul(a3T, b2, transpose_a=True, transpose_b=False)
    ta3b2t = bmatmul(a3, b2T, transpose_a=False, transpose_b=True)
    ta3tb2t = bmatmul(a3T, b2T, transpose_a=True, transpose_b=True)

    sess = tf.InteractiveSession()
    print("Test Broadcasting Matmul")
    for i in range(100):

        a2_sample = np.random.uniform(low=-2.0, high=2.0, size=[dim1, dim2])
        a3_sample = np.random.uniform(low=-2.0, high=2.0, size=[batch_size, dim1, dim2])
        b2_sample = np.random.uniform(low=-2.0, high=2.0, size=[dim2, dim1])
        b3_sample = np.random.uniform(low=-2.0, high=2.0, size=[batch_size, dim2, dim1])

        a2T_sample = np.random.uniform(low=-2.0, high=2.0, size=[dim2, dim1])
        a3T_sample = np.random.uniform(low=-2.0, high=2.0, size=[batch_size, dim2, dim1])
        b2T_sample = np.random.uniform(low=-2.0, high=2.0, size=[dim1, dim2])
        b3T_sample = np.random.uniform(low=-2.0, high=2.0, size=[batch_size, dim1, dim2])

        ta2b3_np = numpy_ref_bmatmul(a2_sample, b3_sample, transposeA=False, transposeB=False)
        ta2b3_tf = sess.run(ta2b3, feed_dict={a2: a2_sample, b3: b3_sample})
        assert np.all(np.isclose(ta2b3_tf, ta2b3_np)), "a2 b3 failed"

        ta2tb3_np = numpy_ref_bmatmul(a2T_sample, b3_sample, transposeA=True, transposeB=False)
        ta2tb3_tf = sess.run(ta2tb3, feed_dict={a2T: a2T_sample, b3: b3_sample})
        assert np.all(np.isclose(ta2tb3_tf, ta2tb3_np)), "a2t b3 failed"

        ta2b3t_np = numpy_ref_bmatmul(a2_sample , b3T_sample, transposeA=False, transposeB=True)
        ta2b3t_tf = sess.run(ta2b3t, feed_dict={a2: a2_sample, b3T: b3T_sample})
        assert np.all(np.isclose(ta2b3t_tf, ta2b3t_np)), "a2 b3t failed"

        ta2tb3t_np = numpy_ref_bmatmul(a2T_sample, b3T_sample, transposeA=True, transposeB=True)
        ta2tb3t_tf = sess.run(ta2tb3t, feed_dict={a2T: a2T_sample, b3T: b3T_sample})
        assert np.all(np.isclose(ta2tb3t_tf, ta2tb3t_np)), "a2t b3t failed"


        ta3b2_np = numpy_ref_bmatmul(a3_sample, b2_sample, transposeA=False, transposeB=False)
        ta3b2_tf = sess.run(ta3b2, feed_dict={a3: a3_sample, b2: b2_sample})
        assert np.all(np.isclose(ta3b2_np, ta3b2_tf)), "a3 b2 failed"

        ta3tb2_np = numpy_ref_bmatmul(a3T_sample, b2_sample, transposeA=True, transposeB=False)
        ta3tb2_tf = sess.run(ta3tb2, feed_dict={a3T: a3T_sample, b2: b2_sample})
        assert np.all(np.isclose(ta3tb2_np, ta3tb2_tf)), "a3T b2 failed"

        ta3b2t_np = numpy_ref_bmatmul(a3_sample, b2T_sample, transposeA=False, transposeB=True)
        ta3b2t_tf = sess.run(ta3b2t, feed_dict={a3: a3_sample, b2T: b2T_sample})
        assert np.all(np.isclose(ta3b2t_np, ta3b2t_tf)), "a3 b2T failed"

        ta3tb2t_np = numpy_ref_bmatmul(a3T_sample, b2T_sample, transposeA=True, transposeB=True)
        ta3tb2t_tf = sess.run(ta3tb2t, feed_dict={a3T: a3T_sample, b2T: b2T_sample})
        assert np.all(np.isclose(ta3tb2t_np, ta3tb2t_tf)), "a3T b2T failed"
    print("B Matmul Test Successful")

if __name__ == '__main__':
    test_diag_a_diag_at()
    test_diag_a_diag_bt()
    test_bmatmul()
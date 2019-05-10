import pykitti as pk
import scipy.misc as sp
import warnings
import os
import numpy as np
import matplotlib.pyplot as plt


class KittiData:

    """ATTENTION: ... run away from this code... """
    SEQUENCE_LENGTHS = np.array([4541, 1101, 4661, 801, 271, 2761, 1101, 1101, 4071, 1591, 1201])
    NUM_SEQUENCES = 11

    """Possible modes - from easy to hard (theoretically)"""
    """At time t feed img_l,t and img_l,t+1 -> target (pose_t+1 - pose_t)"""
    KITTI_MODE_TEMPORAL_PAIR = ["next"]
    """At time t feed img_l,t and img_r,t -> target (post_t - post_t-1)"""
    KITTI_MODE_STEREO_CURRENT = ["stereo"]
    """At time t feed img_l, t and img_r,t -> target (pos_t+1 - pos_t)"""
    #KITTI_MODE_STEREO_NEXT = ["next", "stereo"]

    def __init__(self, base_path, seq_length, mode, test_seqs, offset=None):

        self.base_path = base_path
        self.seq_length = seq_length
        self.mode = mode

        self.offset = seq_length if offset is None else offset
        self.start_offset = 0
        self.stereo_img = "stereo" in self.mode

        self.img_size = [50, 150]

        print("start loading")
        obs_dict, pos_dict = self._load_data()
        obs_dict_train, pos_dict_train, obs_dict_test, pos_dict_test = self._split_data(obs_dict, pos_dict, test_seqs)
        if self.stereo_img:
            train_obs_batch_dict = self._batch_obs_stereo(obs_dict_train)
            test_obs_batch_dict = self._batch_obs_stereo(obs_dict_test)
        else:
            train_obs_batch_dict = self._batch_obs_temp(obs_dict_train)
            test_obs_batch_dict = self._batch_obs_temp(obs_dict_test)
        train_pos_batch_dict = self._batch_pos_diff(pos_dict_train)
        test_pos_batch_dict = self._batch_pos_diff(pos_dict_test)

        self.train_obs, self.train_pos = self._get_arrays(train_obs_batch_dict, train_pos_batch_dict)
        self.test_obs, self.test_pos = self._get_arrays(test_obs_batch_dict, test_pos_batch_dict)
        self.num_train_batches = self.train_obs.shape[0]
        print("Got", self.num_train_batches, "sequences of length", seq_length, "for training!")


    def _load_data(self):
        pos_dict = {}
        obs_dict = {}
        for i in range(KittiData.NUM_SEQUENCES):
            cur_seq = '{:0>2}'.format(i)
            cur_data = pk.odometry(self.base_path, sequence=cur_seq)

            poses_gen = cur_data.poses
            left_imgs_gen = cur_data.cam2
            if self.stereo_img:
                right_imgs_gen = cur_data.cam3
            cur_length = KittiData.SEQUENCE_LENGTHS[i]


            poses = np.zeros([cur_length, 3])
            imgs = np.zeros([cur_length] + self.img_size + [6 if self.stereo_img else 3], dtype=np.uint8)

            for j in range(KittiData.SEQUENCE_LENGTHS[i]):
                #if j % 100 == 0:
                #    print(j)
                pose_mat = next(poses_gen)
                poses[j, 0] = pose_mat[0, 3]
                poses[j, 1] = pose_mat[2, 3]
                poses[j, 2] = self._get_orientation(pose_mat[:3, :3])

                img_left = next(left_imgs_gen)
                img_left *= 255
                imgs[j, :, :, :3] = img_left
                if self.stereo_img:
                    img_right = next(right_imgs_gen)
                    img_right *= 255
                    imgs[j, : ,:, 3:] = img_right


            pos_dict[i] = poses
            obs_dict[i] = imgs

        return obs_dict, pos_dict

    def _split_data(self, obs_dict, pos_dict, test_sequences):
        if not isinstance(test_sequences, list):
            test_sequences = [test_sequences]

        test_obs_dict ={}
        test_pos_dict = {}
        for i in test_sequences:
            test_obs_dict[i] = obs_dict.pop(i)
            test_pos_dict[i] = pos_dict.pop(i)

        return obs_dict, pos_dict, test_obs_dict, test_pos_dict

    def _get_arrays(self, obs_batch_dict, pos_batch_dict):
        obs_batch_list = []
        pos_batch_list = []
        assert set(obs_batch_dict.keys()) == set(pos_batch_dict.keys()), \
            "Observations and positions dict have different keys"
        for seq_idx in pos_batch_dict.keys():
            obs = obs_batch_dict[seq_idx]
            pos = pos_batch_dict[seq_idx]
            assert set(obs.keys()) == set(pos.keys()), "Current observation and position have different keys!"
            for batch_idx in obs.keys():
                if self.mode == KittiData.KITTI_MODE_STEREO_CURRENT:
                    obs_batch_list.append(obs[batch_idx])
                    # add zero to front - remove last value
                    pos_batch_list.append(np.concatenate([np.array([[0, 0, 0]]), pos[batch_idx]], 0)[:-1])

                elif self.mode == KittiData.KITTI_MODE_TEMPORAL_PAIR:
                    obs_batch_list.append(obs[batch_idx])
                    pos_batch_list.append(pos[batch_idx])

                else:
                    raise AssertionError("Invalid Mode, needs to be either KittiData.KITTI_MODE_TEMPORAL_PAIR or" +
                                         "KittiData.KITTI_MODE_STEREO_CURRENT")

        return np.stack(obs_batch_list), np.stack(pos_batch_list)

    def _batch_pos_diff(self, pos_dict):
        batch_dict = {}
        for i in pos_dict.keys():
            pos = pos_dict[i]
            seq_dict = {}
            dict_idx = 0
            for j in range(0, KittiData.SEQUENCE_LENGTHS[i] - self.offset, self.offset):
                seq_dict[dict_idx] = pos[j + 1 : j + self.seq_length + 1] - pos[j : j + self.seq_length]
                dict_idx += 1
            batch_dict[i] = seq_dict
        return batch_dict

    def _batch_obs_stereo(self, obs_dict):
        batch_dict = {}
        for i in obs_dict.keys():
            obs = obs_dict[i]
            seq_dict = {}
            dict_idx = 0
            for j in range(0, KittiData.SEQUENCE_LENGTHS[i] - self.offset, self.offset):
                seq_dict[dict_idx] = obs[j: j + self.seq_length]
                dict_idx += 1
            batch_dict[i] = seq_dict
        return batch_dict

    def _batch_obs_temp(self, obs_dict):
        batch_dict = {}
        for i in obs_dict.keys():
            obs = obs_dict[i]
            seq_dict = {}
            dict_idx = 0
            for j in range(0, KittiData.SEQUENCE_LENGTHS[i] - self.offset, self.offset):
                seq_dict[dict_idx] = \
                    np.concatenate([obs[j : j + self.seq_length], obs[j + 1 : j + self.seq_length + 1]], -1)
                dict_idx += 1
            batch_dict[i] = seq_dict
        return batch_dict

    def _get_orientation(self, rot_mat):
        rc3d = rot_mat.dot(np.array([0, 0, 1]))
        return np.arctan2(rc3d[0], rc3d[2])


    """Dataset stuff - currently invalid"""

    #def get_iterator(self, batch_size=-1, shuffle=True):
    #    assert self.use_dataset, "Initialize kitti data object with use_dataset=True to work with iterator"
    #    ds = self.dataset
    #    if shuffle:
    #        ds = ds.shuffle(self.num_seqs)
    #    ds = ds.map(self._parser)
    #    if batch_size > 0:
    #        ds = ds.batch(batch_size)

     #   return ds.make_initializable_iterator()

    #def _prepare_data_dataset(self):
    #    poses_as_list = []
    #    image_paths_as_list = []
    #    for i in range(KittiData.NUM_SEQUENCES):
    #        cur_seq = '{:0>2}'.format(i)
    #        cur_data = pk.odometry(self.base_path, sequence=cur_seq)
    #        cur_poses_as_mat = cur_data.poses
    #        #there needs to be an easier way for that...
    #        num_batches = int(np.floor(KittiData.SEQUENCE_LENGTHS[i] / self.seq_length))
    #        num_elements = self.seq_length * num_batches

    #        cur_poses = np.zeros([num_batches, self.seq_length, 3])
            #uint16 does not work with tensorflow cast to string....
    #        cur_image_paths = np.zeros([num_batches, self.seq_length, 2], dtype=np.int32)
    #        for j in range(num_elements):
    #            batch_idx = int(np.floor(j / self.seq_length))
    #            seq_idx = j % self.seq_length
    #            cur_poses[batch_idx, seq_idx] = next(cur_poses_as_mat)[:3, 3]
    #            cur_image_paths[batch_idx, seq_idx] = np.array([i, j])

    #        poses_as_list.append(cur_poses)
    #        image_paths_as_list.append(cur_image_paths)

     #   poses = np.concatenate(poses_as_list, 0)
     #   image_paths = np.concatenate(image_paths_as_list, 0)

     #   assert poses.shape[0] == image_paths.shape[0] == self.num_seqs, "something wrong"
     #   return image_paths, poses

    #def _parser(self, img_paths, poses):
    #    img_path = tf.constant(os.path.join(base_path, 'sequences'), dtype=tf.string)
    #    folder_name = tf.as_string(img_paths[0, 0], width=2, fill="0")

    #    img_array = tf.TensorArray(dtype=tf.uint8, size=self.seq_length, element_shape=[1] + self.img_size)
    #    loop_init =(tf.constant(0), img_array)
    #    loop_condition = lambda i, _: tf.less(i, self.seq_length)

    #    def loop_body(i, array):
    #        file_name = tf.string_join([tf.as_string(img_paths[i, 1], width=6, fill="0"), ".png"])
    #        file_path = tf.string_join([img_path, folder_name, "image_2", file_name], separator="/")
    #        img_file = tf.read_file(file_path)
    #        img_decoded = tf.expand_dims(tf.image.decode_image(img_file, channels=3), 0)
    #        img_reshaped = tf.cast(tf.image.resize_bilinear(img_decoded, [376, 1241]), tf.uint8)
    #        array.write(i, img_reshaped)
    #        return i+1, array

    #    loop_final = tf.while_loop(loop_condition,
    #                               loop_body,
    #                               loop_init,
    #                               back_prop=False,
    #                               #Todo .. for some reason I get an error if this is > 1, figure out why, fix!
    #                               parallel_iterations=1)

     #   images = loop_final[1].concat()
    #    images = tf.reshape(images, [self.seq_length, 376, 1241, 3])
    #    return images, poses

    def global_resize(self, size, write_path_suffix):
        """
        Resizes the whole kitti dataset found at base path and writes the resized images under write_path_suffix
        :param size: size of images after resizing
        :param write_path_suffix:
        :return:
        """


        assert False, "If you sure you want to use this uncomment this line"

        write_path = os.path.join(base_path, write_path_suffix)
        if not os.path.exists(write_path):
            warnings.warn("Creating write path: " + write_path)
            os.makedirs(write_path)


        for i in range(KittiData.NUM_SEQUENCES):
            print("Start sequence", i, "with", KittiData.SEQUENCE_LENGTHS[i], "elements")
            seq_suffix = '{:0>2}'.format(i)
            cur_write_path = os.path.join(write_path, seq_suffix)
            if not os.path.exists(cur_write_path):
                os.makedirs(cur_write_path)
            path_cam_2 = os.path.join(cur_write_path, "image_2")
            if not os.path.exists(path_cam_2):
                os.makedirs(path_cam_2)
            path_cam_3 = os.path.join(cur_write_path, "image_3")
            if not os.path.exists(path_cam_3):
                os.makedirs(path_cam_3)

            data = pk.odometry(self.base_path, seq_suffix)
            cam2_gen = data.cam2
            cam3_gen = data.cam3
            for j in range(KittiData.SEQUENCE_LENGTHS[i]):
                if j % 100 == 0:
                    print("At image", j)
                img_name = '{:0>6}'.format(j) + ".png"
                cur_path_cam_2 = os.path.join(path_cam_2, img_name)
                cur_path_cam_3 = os.path.join(path_cam_3, img_name)
                img = next(cam2_gen)
                img = sp.imresize(img, size)
                sp.imsave(os.path.join(cur_write_path, cur_path_cam_2), img)

                img = next(cam3_gen)
                img = sp.imresize(img, size)
                sp.imsave(os.path.join(cur_write_path, cur_path_cam_3), img)





if __name__ == '__main__':
    base_path = "/home/philipp/Code/KittiData/"
    data = KittiData(base_path, 50, KittiData.KITTI_MODE_STEREO_CURRENT, test_seqs=[10])

  #  data.poses


#    data.global_resize([50, 150], 'bkf_sized')

#    data_iterator = data.get_iterator(batch_size=5)
#    print("init")
#    sess = tf.InteractiveSession()
#
#    sess.run(data_iterator.initializer)
#    print("initialized")

#    next_batch = data_iterator.get_next()
#    obs = next_batch[0].eval()
#    targets = next_batch[1].eval()
#    plt.imshow(obs[0][5])
#    plt.show()



 #   data = pk.odometry(base_path, sequence="10")

#    poses_as_mat = data.poses
#    pos_vector = []
#    angle_vector = []

    #data.timestamps
#    for pose in poses_as_mat:
#        pos_vector.append(np.array([pose[0, 3], pose[2, 3]]))
#        rot_mat = pose[:3, :3]






#    poses = np.stack(pos_vector)
 #   angles = np.stack(angle_vector)


#    points = np.expand_dims(poses, 1)
#    segments = np.concatenate([points[:-1], points[1:]], axis=1)
#    t = np.arange(len(poses))
    # Create the line collection object, setting the colormapping parameters.
    # Have to set the actual values used for colormapping separately.
#    lc = LineCollection(segments, cmap=plt.cm.autumn, norm=plt.Normalize(0, len(t)))
#    lc.set_array(t)
#    lc.set_linewidth(3)

#    fig1 = plt.figure()
#    plt.gca().add_collection(lc)
#    plt.xlim(np.min(points[:, :, 0]), np.max(points[:, :, 0]))
#    plt.ylim(np.min(points[:, :, 1]), np.max(points[:, :, 1]))

#    ang_plot = np.expand_dims(np.concatenate([np.expand_dims(t, 1), np.expand_dims(angles, 1)], 1), 1)
#    segments = np.concatenate([ang_plot[:-1], ang_plot[1:]], axis=1)
#    lc = LineCollection(segments, cmap=plt.cm.autumn, norm=plt.Normalize(0, len(t)))
#    lc.set_array(t)
#    lc.set_linewidth(3)
#    fig2 = plt.figure()
#    plt.gca().add_collection(lc)
#    plt.xlim(np.min(ang_plot[:, :, 0]), np.max(ang_plot[:, :, 0]))
#    plt.ylim(np.min(ang_plot[:, :, 1]), np.max(ang_plot[:, :, 1]))




    plt.show()
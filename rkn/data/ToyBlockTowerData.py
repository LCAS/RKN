import os
import scipy.misc as sp
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np

class ToyBlockTowerData:

    def __init__(self, train_set_path, test_set_path, img_size=(120, 160), max_seqs=int(1e5), gs=False,
                 simplify_bg=False, seq_length=25, with_gt=False):
        self.img_size = img_size
        self.img_per_trajectory = np.minimum(seq_length, 50)
        self._first_resize = True
        self.with_gt = with_gt

        self.simplify_bg = simplify_bg
        self.gs = gs
        self.header_length = 8

        if self.with_gt:
            self.train_obs, self.train_targets = self.readSet(train_set_path, max_seqs)
            self.test_obs, self.test_targets = self.readSet(test_set_path, max_seqs)

        else:
            self.train_obs = self.readSet(train_set_path, max_seqs)
            self.test_obs = self.readSet(test_set_path, max_seqs)



    def readSet(self, path, max_seqs):
        num_traj = 0
        for _ in os.listdir(path):
            num_traj += 1

        num_traj = np.minimum(num_traj, max_seqs)
        color_channels = 1 if self.gs else 3
        mode = "L" if self.gs else "RGB"
        obs = np.zeros((num_traj, self.img_per_trajectory, self.img_size[0], self.img_size[1], color_channels), dtype=np.uint8)

        if self.with_gt:
            targets = np.zeros((num_traj, self.img_per_trajectory, 18), dtype=np.float32)

        for i, dir in enumerate(os.listdir(path)):
            if i == num_traj:
                break
            cur_dir = os.path.join(path, dir)

            for j in range(self.img_per_trajectory):
                pic_path = os.path.join(cur_dir, "img"+str(j+1)+".png")

                img = self._fit_size(sp.imread(pic_path, mode=mode))
                if self.simplify_bg:
                    img[np.all(img > 150, -1)] = 255 * np.ones(3)
                obs[i, j, :, :, :] = np.expand_dims(img, -1) if self.gs else img

            if self.with_gt:
                with open(os.path.join(cur_dir, "traj.txt")) as traj_file:
                    for j, line in enumerate(traj_file):
                        idx = j - self.header_length
                        if 0 <= idx < self.img_per_trajectory:
                            coord_list = line.split(";")
                            assert len(coord_list) == 18
                            for k in range(18):
                                num_as_list = coord_list[k].split(',')
                                assert len(num_as_list) <= 2
                                if len(num_as_list) == 2:
                                    targets[i, idx, k] = float(num_as_list[0] + "." + num_as_list[1])
                                elif len(num_as_list) == 1:
                                    targets[i, idx, k] = float(num_as_list[0])
        return (obs, targets) if self.with_gt else obs

    def get_train_data(self):
        return (self.train_obs, self.train_targets) if self.with_gt else self.train_obs

    def get_test_data(self):
        return (self.test_obs, self.test_targets) if self.with_gt else self.test_obs

    def _fit_size(self, img):
        if img.shape[0] != self.img_size[0] or img.shape[1] != self.img_size[1]:
            if self._first_resize:
                print("Image size does not match, resizing to", self.img_size)
                self._first_resize = False
            img = sp.imresize(img, self.img_size)
        return img

if __name__ == '__main__':
    path_prefix = '/home/philipp/RKN/toyBlockData'
    train_path = path_prefix + '/train'
    test_path = path_prefix + '/test'
    jbd = ToyBlockTowerData(train_path, test_path, img_size=(60, 80), max_seqs=50, simplify_bg=False, seq_length=15,
                            with_gt=True)
    print("read data")

    train_obs, train_targets = jbd.get_train_data()
    test_obs, test_targets = jbd.get_test_data()

    labels = ["box 1 pos x", "box 1 pos y", "box 1 pos z", "box 2 pos x", "box 2 pos y", "box 2 pos z", "box 3 pos x", "box 3 pos y", "box 3 pos z"]

    #img[np.all(img > 120, -1)] = 160 * np.ones(3)
    fig = plt.figure(figsize=(9, 54))
    idx = [0,1,2, 12,13,14, 6,7,8]
    for j in range(9) :
        plt.subplot(9, 2, 2*j+1)
        for i in range(50):
            plt.plot(train_targets[i, :, idx[j]])
        plt.title("train " + labels[j])
        plt.subplot(9, 2, 2*j+2)
        for i in range(50):
            plt.plot(test_targets[i, :, idx[j]])
        plt.title("test " + labels[j])
#    for i in range(10):

        #plt.plot(target[i, :seq_length[i], 0], target[i, :seq_length[i], 1])


    def save_fig(fig, path, name, dpi=1200):
        if not os.path.exists(path):
            os.makedirs(path)
        # mp.rcParams['axes.linewidth'] = .2
        #mp.rcParams['lines.linewidth'] = .2
        # mp.rcParams['patch.linewidth'] = .2
        fig.savefig(path + '/' + name + '.pdf', format='pdf', dpi=dpi)
    save_fig(fig, "dummy", "all_traj", dpi=400)
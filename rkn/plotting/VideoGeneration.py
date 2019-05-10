import numpy as np
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as anim

class VideoGeneration:

    def __init__(self, base_path, base_path_ref=None, write_n=20):
        self._base_path = base_path
        self._write_n = write_n
        self._base_path_ref = base_path_ref




    def save_vid_to(self, inputs, predictions, targets, folder):
        full_path = os.path.join(self._base_path, folder)
        self._create_if_not_exists(full_path)
        np.save(os.path.join(full_path, "inputs.npy"), inputs[:self._write_n])
        np.save(os.path.join(full_path, "preds.npy"), predictions[:self._write_n])
        np.save(os.path.join(full_path, "targets.npy"), targets[:self._write_n])

    def create_vid(self, folder, max_vids=-1):
        full_path = os.path.join(self._base_path, folder)
        inputs, predictions, targets = self._load_vid_data(full_path)

        if self._base_path_ref is not None:
            full_path_ref = os.path.join(self._base_path_ref, folder)
            _, ref_predictions, ref_targets = self._load_vid_data(full_path_ref)
        else:
            ref_predictions = ref_targets = None



        for i in range(len(inputs) if max_vids < 0 else np.minimum(max_vids, len(inputs))):
            self._create_vid(inputs[i], predictions[i], targets[i], name=os.path.join(full_path, "vid" + str(i) + ".mp4"),
                             ref_predictions=ref_predictions[i] if ref_predictions is not None else None,
                             ref_targets=ref_targets[i] if ref_targets is not None else None)



    def _create_vid(self, inputs, predictions, targets, name,
                    ref_predictions=None, ref_targets=None):
        cols = 2  if ref_predictions is None else 3
        fig = plt.figure(figsize=[10 * cols + 2, 10])
        predictions = self._to_angle(predictions)
        targets = self._to_angle(targets)
        if ref_predictions is not None:
            ref_predictions = self._to_angle(ref_predictions)
            ref_targets = self._to_angle(ref_targets)



        ax = plt.subplot(1, cols, 1)
        ax.set_aspect('equal')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        img_input = plt.imshow(inputs[0, ::-1,:,0], cmap="gray", interpolation="none")

        pred_plots = []
        target_plots = []
        if ref_predictions is not None:
            ref_pred_plots = []
            ref_target_plots = []


        x_pts = np.arange(len(predictions))
        num_dims = predictions.shape[1]

        for i in range(num_dims):
            ax = plt.subplot(num_dims, cols, 2 +  cols * i)
            print("normal", cols + cols * i)
            tmp, = ax.plot(x_pts, targets[:, i], color='g')
            target_plots.append(tmp)
            tmp, = ax.plot(x_pts, predictions[:, i], color='b')
            pred_plots.append(tmp)
            plt.xlim(0, len(predictions))
            plt.ylim(-np.pi, np.pi)
            ax.set_aspect(len(predictions) / (2 * np.pi * num_dims))
            if i == 0:
                plt.title("RKN", fontsize=24)
            if ref_predictions is not None:
                print("ref", cols + cols * i + 1)
                ax = plt.subplot(num_dims, cols, cols + cols * i)
                tmp, = ax.plot(x_pts, ref_targets[:, i], color='g')
                ref_target_plots.append(tmp)
                tmp, = ax.plot(x_pts, ref_predictions[:, i], color='b')
                ref_pred_plots.append(tmp)
                plt.xlim(0, len(ref_predictions))
                plt.ylim(-np.pi, np.pi)
                ax.set_aspect(len(ref_predictions) / (2 * np.pi * num_dims))
                if i == 0:
                    plt.title("LSTM", fontsize=24)
        #ax.legend(["Target", "Prediction"])

      #  ax = plt.subplot(2, 2, 4)
      #  targ2, = ax.plot(targets[:, 1], color='g')
      #  pred2, = ax.plot(predictions[:, 1], color='b')
        fig.tight_layout()
        def update(n):
            tmp = inputs[n, ::-1, :, 0]
            img_input.set_data(tmp)

            for i in range(num_dims):
                pred_plots[i].set_data(x_pts[:n], predictions[:n, i])
                target_plots[i].set_data(x_pts[:n], targets[:n, i])
                if ref_predictions is not None:
                    ref_pred_plots[i].set_data(x_pts[:n], ref_predictions[:n, i])
                    ref_target_plots[i].set_data(x_pts[:n], ref_targets[:n, i])
          #  pred2.set_data(x_pts[:n], predictions[:n, 1])
          #  targ2.set_data(x_pts[:n], targets[:n, 1])


        ani = anim.FuncAnimation(fig, update, len(predictions))
        writer = anim.writers['ffmpeg'](fps=2)
        ani.save(name, writer=writer)

    def _load_vid_data(self, folder):
        inputs = np.load(os.path.join(folder, "inputs.npy"))
        predictions = np.load(os.path.join(folder, "preds.npy"))
        targets = np.load(os.path.join(folder, "targets.npy"))
        return inputs, predictions, targets

   # def _generate_vid(self, inputs, predictions, targets, folder):


    def _to_angle(self, sin_cos):

        sin = np.clip(sin_cos[:, 0::2], a_min=-1, a_max=1)
        cos = np.clip(sin_cos[:, 1::2], a_min=-1, a_max=1)
        angles = np.zeros(shape=sin.shape)
        for i in range(sin.shape[1]):
            cur = np.arccos(cos[:, i])
            cur [np.arcsin(sin[:, i]) < 0] *= -1
            angles[:, i] = cur
        angles[:, 0] = angles[:, 0]  % (2  *np.pi) -np.pi


        return angles




    def _create_if_not_exists(self, path):
        if not os.path.exists(path):
            os.makedirs(path)



if __name__ == "__main__":
    base_path = "/home/philipp/RKN/vid_data/20180712_0027_45_rkn_Ex_ql_rknc_decMode_lin_lod_100_1/vid_data_model2_quad_rknc"
    ref_path_small = "/home/philipp/RKN/vid_data/20180712_0027_39_rkn_Ex_ql_lstm_decMode_nonlin_lod_25_2/vid_data_model1_quad_lstm"
    ref_path_large = "/home/philipp/RKN/vid_data/20180712_0027_42_rkn_Ex_ql_lstm_decMode_nonlin_lod_100_1/vid_data_model1_quad_lstm"
    vid_gen = VideoGeneration(base_path=base_path, base_path_ref=ref_path_large)#"../vid_data_quad" , base_path_ref="../vid_data_quad_lstm")
    vid_gen.create_vid("iter7", max_vids=1)

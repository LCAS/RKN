import os, warnings
import scipy.misc as sp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as anim

#Todo this needs to be tested...

def save_vid_data(targets, inputs, predictions, path):
    """Saves the images needed to create the sequence
    :param targets: sequence of target images
    :param inputs: sequence of input images
    :param predictions: sequence of predictions
    :param path: path to save underh
    :return:
    """
    path = os.path.join(path)
    if not os.path.exists(path):
        warnings.warn('Path ' + path + ' not found - creating')
        os.makedirs(path)

    for i in range(len(targets)):
        sp.imsave(os.path.join(path, 't'+str(i)+".png"), targets[i])
        sp.imsave(os.path.join(path, 'p'+str(i)+".png"), predictions[i])
        if i < len(inputs):
            sp.imsave(os.path.join(path, 'i'+str(i)+".png"), inputs[i])

def generate_vid(targets, inputs, predictions, path, vid_name, title=None, framerate=20):
    """Tries to generate video out of data, if this fails the images are saved instead an the video can be created later
    by running this file (see below)
    :param targets: sequence of target images
    :param inputs: sequence of input images
    :param predictions: sequence of predictions
    :param path: path to save under
    :param vid_name: name of the file to save under
    :param title: Title displayed in the video, if None then no title is displayed
    :param framerate: framerate of the create video (default: 20 fps)
    :param sigma:
    :return:
    """
    try:
        _generate_vid_internal(targets, inputs, predictions, path, vid_name, title, framerate)
    except Exception:
        save_path = "saved_vid_data_" + vid_name
        print("Video generation failed, saving images under, '"+save_path+"' instead")
        save_vid_data(targets, inputs, predictions, path=save_path)

def _generate_vid_internal(targets, inputs, predictions, path, vid_name, title=None, framerate=20):
    """see super"""
    if not os.path.exists(path):
        warnings.warn('Path ' + path + ' not found - creating')
        os.makedirs(path)
    print("Generating video, with slow motion factor", slowmo_fact)
    dpi = 100
    fig = plt.figure(figsize=[6, 2])
    if title is not None:
        plt.title(title)
    cur_axes = plt.gca()
    cur_axes.axes.get_xaxis().set_visible(False)
    cur_axes.axes.get_yaxis().set_visible(False)
    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    # remove all the ticks and directly label each bar with respective value
    plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='on')

    ax = fig.add_subplot(131)
    ax.set_aspect('equal')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_title("Target Observation")
    im_t = ax.imshow(targets[0], cmap="gray", interpolation='none')
    ax = fig.add_subplot(132)
    ax.set_aspect('equal')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_title("Model Input")
    im_i = ax.imshow(inputs[0], cmap="gray", interpolation='none')
    ax = fig.add_subplot(133)
    ax.set_aspect('equal')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_title("Prediction")
    im_p = ax.imshow(predictions[0], cmap="gray", interpolation='none')

    plt.tight_layout()

    def update_img(n):
        tmp = targets[n]
        im_t.set_data(tmp)
        tmp = inputs[n]
        im_i.set_data(tmp)
        tmp = predictions[n]
        im_p.set_data(tmp)

    ani = anim.FuncAnimation(fig, update_img, len(predictions))
    writer = anim.writers['ffmpeg'](fps=framerate)
    ani.save(path + '/' + vid_name + '.mp4', writer=writer, dpi=dpi)
    plt.close()
    return ani

def load_vid_data(base_path, img_size, length, input_length=None, mode="L"):
    """Loads data saved by the save_vid_data
    :param base_path: Path of to the folder the images are in
    :param img_size: size of the images
    :param length: length of the prediction and target sequence
    :param input_length: length of the input sequence (default: same as target and prediction sequence i.e. length)
    :param mode: Either "L" for grayscale images or "RGB" for, well, RGB images
    :return: Sequences of target, input and prediction images
    """
    img_size = img_size + [3] if mode == "RGB" else img_size
    input_length = input_length if input_length is not None else length

    predictions = np.zeros([length] + img_size, dtype=np.uint8)
    targets = np.zeros([length] + img_size, dtype=np.uint8)
    inputs = 255 * np.ones([length] + img_size, dtype=np.uint8)

    for i in range(length):
        predictions[i] = sp.imread(os.path.join(base_path, "p" + str(i) + ".png"), mode=mode)
        targets[i] = sp.imread(os.path.join(base_path, "t" + str(i) + ".png"), mode=mode)
        if i < input_length:
            inputs[i] = sp.imread(os.path.join(base_path, "i" + str(i) + ".png"), mode=mode)
    return targets, inputs, predictions

if __name__ == '__main__':
    """This loads data saved by save_vid_data and creates a video. Can be used if the video can not be created on the
    same machine the model is run on, e.g due to missing libraries
    (For users from TU Darmstadt: this is the case for the Lichtenberg Cluster)"""
    # Data Loading parameters
    base_path = ""
    img_size=[48, 48]
    length = 150
    input_length = 150
    mode = "L"

    # Video generation Parameters
    vid_path = ""
    vid_title = ""
    slowmo_fact = 1
    sigma = 0.0

    targets, inputs, predictions = load_vid_data(base_path, img_size, length, input_length, mode)
    generate_vid(targets, inputs, predictions, vid_path, vid_title, slowmo_fact, sigma)


import os
import numpy as np
import pickle

"""From the PGM-project"""

COINS = "BTC ETH LTC XMR DASH"


class Normalizer:
    norm_methods = dict(mean_zero_unit_std=lambda _x: (np.mean(_x), np.std(_x)),
                        siraj_norm=lambda _x: (_x[0], np.maximum(_x[0], 1e-5)),
                        min_zero_max_one=lambda _x: (np.min(_x), np.max(_x - np.min(_x))))

    def __init__(self, norm_method, x):
        stats_fun = Normalizer.norm_methods[norm_method] if type(norm_method) is str else norm_method
        self.sub_val, self.div_val = stats_fun(x)

    def normalize(self, x):
        return (x-self.sub_val) / self.div_val

    def un_normalize(self, x):
        return x*self.div_val + self.sub_val

    def __call__(self, x):
        return self.normalize(x)

    @staticmethod
    def apply(norm_method, x):
        stats_fun = Normalizer.norm_methods[norm_method] if type(norm_method) is str else norm_method
        sub_val, div_val = stats_fun(x)
        return (x-sub_val)/div_val


def gauss_pdf(x, var):
    return np.exp(-.5 * (x / var) ** 2.)


def gauss_filter(data, filter_len=5, var=.1):
    if len(data.shape) == 2:
        return np.hstack([gauss_filter(data[:, i], filter_len=filter_len, var=var) for i in range(data.shape[1])])
    weights = gauss_pdf(np.linspace(-1, 1, filter_len), var)
    weights = weights / np.sum(weights)
    # expand the data left and right, otherwise "same" convolutions will assume zeros
    d = np.hstack([data[:filter_len][::-1], data, data[-filter_len:][::-1]])
    return np.convolve(d, weights, "same").reshape(-1, 1)[filter_len:-filter_len]


class DataManager:
    def __init__(self, data_path="dataset_features_btc_relative.pkl", feature_dims=(0, ), target_coin="BTC",
                 data_smoothing=True, input_length=100, batch_size=512, perc_test=0.1,
                 norm_method="mean_zero_unit_std",
                 prediction_offset=6, target_window_size=6, boolean_targets=False):
        assert os.path.isfile(data_path)
        self.data_path, self.feature_dims, self.smoothing = data_path, feature_dims, data_smoothing
        self.input_length, self.batch_size = input_length, batch_size
        self.prediction_offset = prediction_offset
        self.target_coin_idx = COINS.index(target_coin)
        self.norm_method = norm_method
        self.features = self.load_features()
        self.boolean_targets = boolean_targets
        self.train_features, self.train_targets, self.test_features, self.test_targets = \
            self.get_training_test_data(self.features, prediction_offset, target_window_size, perc_test,
                                        boolean_targets)
        self.train_batch_gen = self.batch_generator(self.train_features, self.train_targets)
        self.test_batch_gen = self.batch_generator(self.test_features, self.test_targets)

    @property
    def train_data_batched(self):
        return self.get_batches(self.train_features, self.train_targets)

    @property
    def test_data_batched(self):
        return self.get_batches(self.test_features, self.test_targets)

    def load_features(self):
        data_list = []
        with open(self.data_path, 'rb') as f:
            all_data = pickle.load(f)
            # removed ZEC and BCH from list here since they have to few data points!
            for coin in COINS.split():
                cur_data = all_data[coin][:, 1:]
                data_list.append(cur_data)

            min_length = min([len(x) for x in data_list])
            features = np.concatenate(np.array([x[-min_length:, self.feature_dims] for x in data_list]), -1)
            btc, altc = features[:, :1], features[:, 1:]
            relative_values = Normalizer.apply(self.norm_method, btc / np.sum(altc, axis=1).reshape(-1, 1))
            return np.concatenate([btc, altc, relative_values], axis=1)

    def get_mean_window_targets(self, features, window_offset, window_size, boolean=False):
        dt, w = window_offset, window_size
        n = features.shape[0] - w - dt
        windows = [features[i+dt-int(w/2):i+dt+int(w/2), self.target_coin_idx] for i in range(n)]
        targets = [np.mean(w) for w in windows]
        if boolean:
            last_values = [features[i, self.target_coin_idx] for i in range(n)]
            targets = [int(t > l) for t, l in zip(targets, last_values)]
        return np.array(targets).reshape(-1, 1)

    def get_training_test_data(self, features, prediction_offset, window_size, perc_test, boolean_targets):
        features = gauss_filter(features, filter_len=10, var=1) if self.smoothing else features
        targets = self.get_mean_window_targets(features, prediction_offset, window_size, boolean_targets)
        features = np.array(features[:len(targets):])
        n_test = int(len(features)*perc_test)
        return features[:-n_test], targets[:-n_test], features[-n_test:], targets[-n_test:]

    def get_batch_entry(self, features, targets, i):
        feature_set = features[i:i + self.input_length, :]
        target_set = targets[i:i + self.input_length, :]
        return self._get_batch_entry_internal(feature_set, target_set)

    def _get_batch_entry_internal(self, feature_set, target_set):
        # normalize bitcoin (value in USD) and others (value in BTC) separately, use target coin stats for target norm
        btc, altc, relative_values = feature_set[:, :1], feature_set[:, 1:-1], feature_set[:, -1:]
        btc_normalizer, altc_normalizer = Normalizer(self.norm_method, btc), Normalizer(self.norm_method, altc)
        # normalize targets with the same statistics as target coin
        if self.boolean_targets:
            return (np.hstack([btc_normalizer(btc), altc_normalizer(altc), relative_values]), target_set)
        else:
            return(np.hstack([btc_normalizer(btc), altc_normalizer(altc), relative_values]),
                  (btc_normalizer if self.target_coin_idx == 0 else altc_normalizer)(target_set))

    def get_batches(self, features, targets):
         return map(np.array, zip(*[self.get_batch_entry(features, targets, i)
                                    for i in range(len(features)-self.input_length)]))

    def batch_generator(self, features, targets):
        batch_features, batch_targets = self.get_batches(features, targets)

        while True:
            indices = np.random.choice(len(batch_features), size=self.batch_size)
            yield batch_features[indices], batch_targets[indices]


if __name__ == "__main__":
    DATA_PATH = "/home/philipp/Code/dataset_features_btc_relative.pkl"
    data = DataManager(data_path=DATA_PATH, batch_size=512, input_length=512, data_smoothing=True, prediction_offset=24,
                       target_window_size=6)

    from matplotlib import pyplot as plt
    while True:
        input_batch, target_batch = next(data.train_batch_gen)
        plt.plot(input_batch[0][:, 0], label="btc", linewidth=5, c="blue")
        plt.plot(input_batch[0][:, 1], label="altcoins", c="gray", linewidth=2)
        plt.plot(input_batch[0][:, 1:-1], c="gray", linewidth=2)
        plt.plot(input_batch[0][:, -1], label="altcoins, relative_value", c="orange", linewidth=5)
        plt.plot(list(range(len(input_batch[0][:, 0]), len(input_batch[0][:, 0]) + data.prediction_offset)),
                 target_batch[0][-data.prediction_offset:], label="targets", c="green", linewidth=5)
        plt.legend()
        plt.show()


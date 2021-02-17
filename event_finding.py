from util import average_around, thresholding_algo
import numpy as np
from matplotlib import pyplot as plt


def get_first_trough_index(f, last=False, debug=False):
    """ Tries to find stationary/return point of trace including pulling and 
    relaxation. Looks at standard deviation of a running mean and signals at
    abrupt drops.
    """
    stds = []
    for i in range(25, len(f) - 25):
        std = average_around(f, i, half_n=25)["std"]
        if last:
            stds.insert(0, std)
        else:
            stds.append(std)

    div = 4
    peaksign = thresholding_algo(stds, int(len(f) / div), 4., 0)["signals"]
    while min(peaksign) > -1:
        div = div + 1
        peaksign = thresholding_algo(stds, int(len(f) / div), 4., 0)["signals"]
    if debug:
        print(div)
    if last:
        return len(f) - np.arange(25, len(stds) + 25)[peaksign <= -1][0]
    return np.arange(25, len(stds) + 25)[peaksign <= -1][0]


def find_transitions(y: np.ndarray, noise_estimation_window: tuple = None):
    """ Tries to find unfolding events by looking for negative outliers in
    force change that exceed by a factor of background noise.
    Thanks goes out to Christopher Battle for providing the original code.
    """
    EPS = 1e-4  # SNR stabilization factor

    # Magic numbers
    SNR_SCALE_FACTOR = 10
    MIN_OUTLIER_FACTOR = 1.5
    MAX_OUTLIER_FACTOR = 4.5
    MIN_PERCENTILE = 10

    # Get noise estimation window
    if noise_estimation_window is None:
        end_slice = max(int(len(y)/10), 3)
        s = slice(0, end_slice)
    else:
        s = slice(*noise_estimation_window)

    # Calculate outlier threshold
    snr = (y.max() - y.min()) / (y[s].std() + EPS)
    outlier_factor = min(max(snr/SNR_SCALE_FACTOR, MIN_OUTLIER_FACTOR),
                         MAX_OUTLIER_FACTOR)

    # Find outliers that deviate below the threshold (since force transitions are always negative in slope)
    dy = np.diff(y)
    low_percentile = np.nanpercentile(dy, MIN_PERCENTILE)
    median_low_diff = np.nanmedian(dy) - low_percentile
    outlier_threshold = low_percentile - outlier_factor * median_low_diff

    where = np.where(dy < outlier_threshold)[0]
    if len(where) > 1:
        for i in reversed(range(1, len(where))):
            if where[i] - where[i - 1] <= 5:  # 5 is arbitrary guess
                where = np.delete(where, i)

    return where, outlier_threshold


def plot_events(fdcurves):
    """ Constructs a plot for each member of fdcurves which highlights events of
    interest and targets for fitting.
    """
    plt.figure(figsize=(8, 24))
    i = 1
    for key, val in fdcurves.items():
        fdata = val['force_data']
        unfolds = list(val['unfolds'])
        unfolds.insert(0, 0)
        legs = val['legs']
        top = val['top']
        plt.subplot(len(fdcurves), 1, i)
        plt.plot(np.arange(len(fdata)), fdata, c='tab:blue')
        for j in range(1, len(unfolds)):
            #plt.plot(np.arange(unfolds[j-1]+5, unfolds[j]),
                     #fdata[unfolds[j-1]+5:unfolds[j]])
            plt.plot(np.arange(unfolds[j], unfolds[j]+5),
                     fdata[unfolds[j]:unfolds[j]+5], c='tab:orange')

        for leg in legs:
            plt.plot(np.arange(len(fdata))[leg],
                     fdata[leg], c='tab:green')
        plt.plot(np.arange(top[0], top[1]), fdata[top[0]:top[1]], c='tab:red')

        i += 1

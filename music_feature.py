import librosa
import numpy as np


def make_feature(path):
    y, sr = librosa.load(path)
    y, _ = librosa.effects.trim(y)

    # Fix length to exactly 3s
    target_len = sr * 3
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]

    y = librosa.util.normalize(y)

    mfcc      = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    centroid  = librosa.feature.spectral_centroid(y=y, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff   = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr       = librosa.feature.zero_crossing_rate(y)
    rms       = librosa.feature.rms(y=y)
    contrast  = librosa.feature.spectral_contrast(y=y, sr=sr)

    features = np.hstack([
        mfcc.mean(axis=1), mfcc.std(axis=1),   # 13 + 13
        centroid.mean(),   centroid.std(),       # 1  + 1
        bandwidth.mean(),  bandwidth.std(),      # 1  + 1
        rolloff.mean(),    rolloff.std(),        # 1  + 1
        zcr.mean(),        zcr.std(),            # 1  + 1
        rms.mean(),        rms.std(),            # 1  + 1
        contrast.mean(axis=1), contrast.std(axis=1),  # 7 + 7
    ])

    return features  # 50 features total

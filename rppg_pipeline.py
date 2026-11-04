"""
DeepFakesON-Phys: Physiological Signal Processing Pipeline
Extracts remote Photoplethysmography (rPPG), Blood Volume Pulse (BVP),
and Heart Rate (HR) from facial video sequences.

Based on BiDAlab research and DeepPhys Convolutional Attention Network (CAN).
"""

import numpy as np
try:
    from scipy import signal
except ImportError:
    signal = None



class RPPGExtractor:
    """
    Extracts physiological Blood Volume Pulse (BVP) from facial video frames
    using skin chrominance (CHROM) and Plane-Orthogonal-to-Skin (POS) algorithms.
    """
    def __init__(self, fps=30.0, low_freq=0.75, high_freq=2.5):
        """
        :param fps: Video frame rate (frames per second)
        :param low_freq: Lower bound of cardiac frequency in Hz (0.75 Hz = 45 BPM)
        :param high_freq: Upper bound of cardiac frequency in Hz (2.5 Hz = 150 BPM)
        """
        self.fps = fps
        self.low_freq = low_freq
        self.high_freq = high_freq

    def extract_skin_rois(self, frame, face_box=None):
        """
        Extract primary skin regions of interest (ROI) where blood perfusion is strongest:
        Forehead, Left Cheek, and Right Cheek.
        """
        h, w, _ = frame.shape
        if face_box is None:
            # Default center face assumption if no bounding box provided
            x, y, fw, fh = int(w * 0.25), int(h * 0.15), int(w * 0.5), int(h * 0.7)
        else:
            x, y, fw, fh = face_box

        # Forehead: Upper 15% to 35% of face, center 50% width
        fh_roi = frame[int(y + 0.12 * fh):int(y + 0.32 * fh), int(x + 0.25 * fw):int(x + 0.75 * fw)]
        
        # Left Cheek: 45% to 70% height, 15% to 40% width
        lc_roi = frame[int(y + 0.45 * fh):int(y + 0.70 * fh), int(x + 0.12 * fw):int(x + 0.38 * fw)]
        
        # Right Cheek: 45% to 70% height, 60% to 85% width
        rc_roi = frame[int(y + 0.45 * fh):int(y + 0.70 * fh), int(x + 0.62 * fw):int(x + 0.88 * fw)]

        rois = [r for r in [fh_roi, lc_roi, rc_roi] if r.size > 0]
        
        if not rois:
            # Fallback to center region
            center = frame[int(y + 0.2 * fh):int(y + 0.7 * fh), int(x + 0.2 * fw):int(x + 0.8 * fw)]
            return np.mean(center, axis=(0, 1))

        # Spatial average of RGB channels across ROIs
        means = [np.mean(r, axis=(0, 1)) for r in rois]
        return np.mean(means, axis=0) # returns [B, G, R]

    def chrom_rppg(self, rgb_signals):
        """
        Chrominance-based rPPG method (CHROM, De Haan & Jeanne, 2013).
        Overcomes motion artifacts by projecting RGB into color difference spaces.
        """
        # rgb_signals is shape (N, 3) where columns are [R, G, B]
        R = rgb_signals[:, 0]
        G = rgb_signals[:, 1]
        B = rgb_signals[:, 2]

        # Normalized color signals
        Rn = R / (np.mean(R) + 1e-6)
        Gn = G / (np.mean(G) + 1e-6)
        Bn = B / (np.mean(B) + 1e-6)

        # Chrominance signals:
        # Xs = 3Rn - 2Gn
        # Ys = 1.5Rn + Gn - 1.5Bn
        Xs = 3.0 * Rn - 2.0 * Gn
        Ys = 1.5 * Rn + Gn - 1.5 * Bn

        # Standard deviation ratio for adaptive weighting
        std_x = np.std(Xs)
        std_y = np.std(Ys)
        alpha = (std_x / (std_y + 1e-6))

        # BVP signal: S = Xs - alpha * Ys
        bvp = Xs - alpha * Ys
        return bvp

    def pos_rppg(self, rgb_signals, window_len=32):
        """
        Plane-Orthogonal-to-Skin (POS) rPPG method (Wang et al., 2017).
        Returns an overlap-averaged pulse estimate for all covered frames.
        """
        rgb_signals = np.asarray(rgb_signals, dtype=float)
        N = len(rgb_signals)
        if N == 0:
            return np.asarray([], dtype=float)
        window_len = max(3, min(int(window_len), N))
        H = np.zeros(N, dtype=float)
        counts = np.zeros(N, dtype=float)

        # Include the final possible window and average overlapping estimates.
        for t in range(N - window_len + 1):
            window = rgb_signals[t:t + window_len]
            mean_c = np.mean(window, axis=0) + 1e-6
            Cn = window / mean_c
            S1 = Cn[:, 1] - Cn[:, 2]
            S2 = Cn[:, 1] + Cn[:, 2] - 2.0 * Cn[:, 0]
            alpha = np.std(S1) / (np.std(S2) + 1e-6)
            h = S1 + alpha * S2
            h -= np.mean(h)
            H[t:t + window_len] += h
            counts[t:t + window_len] += 1.0

        covered = counts > 0
        H[covered] /= counts[covered]
        return H

    def bandpass_filter(self, signal_data):
        """
        Butterworth bandpass filter restricted to typical human physiological heart rate:
        0.75 Hz to 2.5 Hz (45 to 150 BPM).
        """
        nyquist = 0.5 * self.fps
        low = self.low_freq / nyquist
        high = self.high_freq / nyquist

        if signal is None:
            # Fallback detrending if scipy is not installed
            return signal_data - np.mean(signal_data)

        # Guard against invalid filter ranges
        low = max(0.01, min(low, 0.9))
        high = max(low + 0.05, min(high, 0.99))

        b, a = signal.butter(4, [low, high], btype='bandpass')
        filtered = signal.filtfilt(b, a, signal_data)
        return filtered

    def compute_spectral_hr(self, bvp_signal):
        """
        Performs Fast Fourier Transform (FFT) on filtered BVP signal to determine:
        1. Dominant Heart Rate in Beats Per Minute (BPM)
        2. Signal-to-Noise Ratio (SNR) in dB
        3. Power Spectral Density distribution
        """
        n = len(bvp_signal)
        if n < 16:
            return 72.0, 0.0, [], []

        # Zero-padding for higher spectral resolution
        n_fft = max(1024, 2 ** int(np.ceil(np.log2(n) + 1)))
        
        # Apply Hanning window
        windowed = bvp_signal * np.hanning(n)
        fft_vals = np.abs(np.fft.rfft(windowed, n=n_fft))
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / self.fps)

        # Restrict to physiological range [0.75, 2.5] Hz
        mask = (freqs >= self.low_freq) & (freqs <= self.high_freq)
        phys_freqs = freqs[mask]
        phys_psd = fft_vals[mask] ** 2

        if len(phys_psd) == 0 or np.sum(phys_psd) < 1e-6:
            return 72.0, 0.0, [], []

        # Peak frequency
        peak_idx = np.argmax(phys_psd)
        peak_freq = phys_freqs[peak_idx]
        hr_bpm = peak_freq * 60.0

        # Calculate SNR around peak (+/- 0.15 Hz harmonic window)
        f_peak = peak_freq
        signal_mask = (phys_freqs >= f_peak - 0.15) & (phys_freqs <= f_peak + 0.15)
        signal_power = np.sum(phys_psd[signal_mask])
        noise_power = np.sum(phys_psd[~signal_mask]) + 1e-6
        snr_db = 10.0 * np.log10(signal_power / noise_power)

        # Downsample spectrum for web visualization (e.g. 50 points)
        step = max(1, len(phys_freqs) // 50)
        spectrum_freqs = (phys_freqs[::step] * 60.0).tolist()
        spectrum_power = (phys_psd[::step] / (np.max(phys_psd) + 1e-6)).tolist()

        return round(float(hr_bpm), 1), round(float(snr_db), 2), spectrum_freqs, spectrum_power


# DeepPhys PyTorch Architecture Simulation / Definition
try:
    import torch
    import torch.nn as nn

    class AttentionMask(nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x):
            xsum = torch.sum(x, dim=(2, 3), keepdim=True)
            norm = x / (xsum + 1e-7)
            h, w = x.shape[2], x.shape[3]
            return norm * (h * w)

    class DeepPhysCAN(nn.Module):
        """
        Convolutional Attention Network (CAN) based on Chen & McDuff (DeepPhys, 2018)
        and adapted by BiDAlab for DeepFakesON-Phys.
        Consists of:
          - Spatial Stream: extracts skin regions and computes attention masks.
          - Motion Stream: processes normalized temporal frame difference (dA/dt).
        """
        def __init__(self):
            super().__init__()
            # Appearance (Spatial) Model
            self.app_conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
            self.app_conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
            self.app_att1 = AttentionMask()
            
            self.app_conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.app_conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
            self.app_att2 = AttentionMask()

            # Motion (Temporal) Model
            self.mot_conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
            self.mot_conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
            self.mot_conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.mot_conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

            # Pooling & Regressor / Classifier
            self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
            self.fc1 = nn.Linear(64 * 18 * 18, 128)
            self.fc_hr = nn.Linear(128, 1)        # Heart rate pulse regression
            self.fc_fake = nn.Linear(128, 2)      # Deepfake classification (Real vs Fake)

        def forward(self, appearance_frame, motion_diff):
            # Spatial Stream
            a1 = torch.relu(self.app_conv2(torch.relu(self.app_conv1(appearance_frame))))
            att1 = self.app_att1(a1)
            a1_pool = self.pool(a1)
            
            a2 = torch.relu(self.app_conv4(torch.relu(self.app_conv3(a1_pool))))
            att2 = self.app_att2(a2)

            # Motion Stream gated by Attention Masks
            m1 = torch.relu(self.mot_conv1(motion_diff))
            m1 = torch.relu(self.mot_conv2(m1))
            m1_gated = m1 * att1
            m1_pool = self.pool(m1_gated)

            m2 = torch.relu(self.mot_conv3(m1_pool))
            m2 = torch.relu(self.mot_conv4(m2))
            m2_gated = m2 * att2
            m2_pool = self.pool(m2_gated)

            feat = m2_pool.view(m2_pool.size(0), -1)
            dense = torch.relu(self.fc1(feat))
            pulse_pred = self.fc_hr(dense)
            fake_logits = self.fc_fake(dense)
            return pulse_pred, fake_logits

except ImportError:
    # PyTorch optional for lightweight installations
    DeepPhysCAN = None

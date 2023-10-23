import numpy as np
import itertools
import scipy.interpolate
import pywt

class SensorDataTransformer:
    def __init__(self, random_seed=None):
        self.transform_func_mapping = {
            0: 'noise_vectorized',
            1: 'scaling_vectorized',
            2: 'rotation_vectorized',
            3: 'negate_vectorized',
            4: 'time_flip_vectorized',
            5: 'channel_shuffle_vectorized',
            6: 'time_shift',
            7: 'amplify_attenuate',
            8: 'add_random_noise',
            9: 'random_phase_shift',
            10: 'spectral_distortion',
            11: 'phase_modulation'
            #6: 'time_segment_permutation',

        }

        self.random_seed = random_seed
        
    
        
    def generate_composite_transform_function_simple(self, transform_funcs):
        def combined_transform_func(X):
            for func in transform_funcs:
                if X.shape[2] == 6:  
                        split_X = np.split(X, 2, axis=-1)  
                        X1 = func(split_X[0])
                        X2 = func(split_X[1])
                        X = np.concatenate([X1,X2], axis=-1)
                else:
                    X = func(X)
            return X
        return combined_transform_func
        
    def get_transform_function(self, transform_indices):
        transform_funcs = []
        for index in transform_indices:
            if index not in self.transform_func_mapping:
                raise ValueError(f"Invalid transform function index: {index}")
            long_name = self.transform_func_mapping[index]
            transform_func = getattr(self, f"{long_name}_transform")
            transform_funcs.append(transform_func)
        return self.generate_composite_transform_function_simple(transform_funcs)

        

    def noise_vectorized_transform(self,X, sigma=0.05):
        """
        Adding random Gaussian noise with mean 0
        """
        noise = np.random.normal(loc=0, scale=sigma, size=X.shape)
        return X + noise

    def scaling_vectorized_transform(self,X, sigma=0.5):
        """
        Scaling by a random factor
        """
        scaling_factor = np.random.normal(loc=1.0, scale=sigma, size=(X.shape[0], 1, X.shape[2]))
        return X * scaling_factor

    def rotation_vectorized_transform(self,X):
        """
        Applying a random 3D rotation
        """
        axes = np.random.uniform(low=-1, high=1, size=(X.shape[0], X.shape[2]))
        angles = np.random.uniform(low=-np.pi, high=np.pi, size=(X.shape[0]))
        matrices = self.axis_angle_to_rotation_matrix_3d_vectorized(axes, angles)

        return np.matmul(X, matrices)

    def axis_angle_to_rotation_matrix_3d_vectorized(self,axes, angles):
        """
        Get the rotational matrix corresponding to a rotation of (angle) radian around the axes

        Reference: the Transforms3d package - transforms3d.axangles.axangle2mat
        Formula: http://en.wikipedia.org/wiki/Rotation_matrix#Axis_and_angle
        """
        axes = axes / np.linalg.norm(axes, ord=2, axis=1, keepdims=True)
        x = axes[:, 0]; y = axes[:, 1]; z = axes[:, 2]
        c = np.cos(angles)
        s = np.sin(angles)
        C = 1 - c

        xs = x*s;   ys = y*s;   zs = z*s
        xC = x*C;   yC = y*C;   zC = z*C
        xyC = x*yC; yzC = y*zC; zxC = z*xC

        m = np.array([
            [ x*xC+c,   xyC-zs,   zxC+ys ],
            [ xyC+zs,   y*yC+c,   yzC-xs ],
            [ zxC-ys,   yzC+xs,   z*zC+c ]])
        matrix_transposed = np.transpose(m, axes=(2,0,1))
        return matrix_transposed

    def negate_vectorized_transform(self,X):
        """
        Inverting the signals
        """
        return X * -1

    def time_flip_vectorized_transform(self,X):
        """
        Reversing the direction of time
        """
        return X[:, ::-1, :]


    def channel_shuffle_vectorized_transform(self,X):
        """
        Shuffling the different channels

        Note: it might consume a lot of memory if the number of channels is high
        """
        channels = range(X.shape[2])
        all_channel_permutations = np.array(list(itertools.permutations(channels))[1:])

        random_permutation_indices = np.random.randint(len(all_channel_permutations), size=(X.shape[0]))
        permuted_channels = all_channel_permutations[random_permutation_indices]
        X_transformed = X[np.arange(X.shape[0])[:, np.newaxis, np.newaxis], np.arange(X.shape[1])[np.newaxis, :, np.newaxis], permuted_channels[:, np.newaxis, :]]
        return X_transformed

    def time_segment_permutation_transform(self,X, num_segments=4):
        """
        Randomly scrambling sections of the signal
        """
        segment_points_permuted = np.random.choice(X.shape[1], size=(X.shape[0], num_segments))
        segment_points = np.sort(segment_points_permuted, axis=1)

        X_transformed = np.empty(shape=X.shape)
        for i, (sample, segments) in enumerate(zip(X, segment_points)):
            # print(sample.shape)
            splitted = np.array(np.split(sample, np.append(segments, X.shape[1])))
            np.random.shuffle(splitted)
            concat = np.concatenate(splitted, axis=0)
            X_transformed[i] = concat
        return X_transformed

    def get_cubic_spline_interpolation(self,x_eval, x_data, y_data):
        """
        Get values for the cubic spline interpolation
        """
        cubic_spline = scipy.interpolate.CubicSpline(x_data, y_data)
        return cubic_spline(x_eval)


    def time_warp_transform(self,X, sigma=0.2, num_knots=4):
        """
        Stretching and warping the time-series
        """
        time_stamps = np.arange(X.shape[1])
        knot_xs = np.arange(0, num_knots + 2, dtype=float) * (X.shape[1] - 1) / (num_knots + 1)
        spline_ys = np.random.normal(loc=1.0, scale=sigma, size=(X.shape[0] * X.shape[2], num_knots + 2))

        spline_values = np.array([self.get_cubic_spline_interpolation(time_stamps, knot_xs, spline_ys_individual) for spline_ys_individual in spline_ys])

        cumulative_sum = np.cumsum(spline_values, axis=1)
        distorted_time_stamps_all = cumulative_sum / cumulative_sum[:, -1][:, np.newaxis] * (X.shape[1] - 1)

        X_transformed = np.empty(shape=X.shape)
        for i, distorted_time_stamps in enumerate(distorted_time_stamps_all):
            X_transformed[i // X.shape[2], :, i % X.shape[2]] = np.interp(time_stamps, distorted_time_stamps, X[i // X.shape[2], :, i % X.shape[2]])
        return X_transformed

    def time_warp_low_cost_transform(self,X, sigma=0.2, num_knots=4, num_splines=150):
        """
        Stretching and warping the time-series (low cost)
        """
        time_stamps = np.arange(X.shape[1])
        knot_xs = np.arange(0, num_knots + 2, dtype=float) * (X.shape[1] - 1) / (num_knots + 1)
        spline_ys = np.random.normal(loc=1.0, scale=sigma, size=(num_splines, num_knots + 2))

        spline_values = np.array([self.get_cubic_spline_interpolation(time_stamps, knot_xs, spline_ys_individual) for spline_ys_individual in spline_ys])

        cumulative_sum = np.cumsum(spline_values, axis=1)
        distorted_time_stamps_all = cumulative_sum / cumulative_sum[:, -1][:, np.newaxis] * (X.shape[1] - 1)

        random_indices = np.random.randint(num_splines, size=(X.shape[0] * X.shape[2]))

        X_transformed = np.empty(shape=X.shape)
        for i, random_index in enumerate(random_indices):
            X_transformed[i // X.shape[2], :, i % X.shape[2]] = np.interp(time_stamps, distorted_time_stamps_all[random_index], X[i // X.shape[2], :, i % X.shape[2]])
        return X_transformed
    
    ########### new transforms 
    def time_shift_transform(self, signal, max_shift=5):
        shift = np.random.randint(-max_shift, max_shift + 1)
        return np.roll(signal, shift)

    def amplify_attenuate_transform(self, signal, min_gain=0.5, max_gain=2.0):
        gain = np.random.uniform(min_gain, max_gain)
        return signal * gain
    
    def add_random_noise_transform(self, signal, noise_std=0.1):
        noise = np.random.normal(0, noise_std, signal.shape)
        return signal + noise
    
    def random_filter_transform(self, X, min_cutoff=0.01, max_cutoff=0.1):
        from scipy.signal import butter, filtfilt
        num_samples, num_timesteps, num_channels = X.shape
        X_filtered = np.empty_like(X)

        for i in range(num_samples):
            for j in range(num_channels):
                cutoff = np.random.uniform(min_cutoff, max_cutoff)
                b, a = butter(4, cutoff)
                X_filtered[i, :, j] = filtfilt(b, a, X[i, :, j])

        return X_filtered
    
    def random_resampling_transform(self, signal, min_resample_factor=0.5, max_resample_factor=2.0, target_length=60):
        from scipy import signal as scipy_signal
        num_samples, num_timesteps, num_channels = signal.shape
        signal_resampled = np.empty((num_samples, target_length, num_channels))

        for i in range(num_samples):
            for j in range(num_channels):
                resample_factor = np.random.uniform(min_resample_factor, max_resample_factor)
                resampled_signal = scipy_signal.resample(signal[i, :, j], int(num_timesteps * resample_factor))

                # Ajustar la longitud al objetivo (target_length)
                if len(resampled_signal) < target_length:
                    signal_resampled[i, :, j] = np.pad(resampled_signal, (0, target_length - len(resampled_signal)))
                else:
                    signal_resampled[i, :, j] = resampled_signal[:target_length]

        return signal_resampled
    
    def random_interpolation_transform(self, signal, num_points=100):
        #random_state = np.random.RandomState(self.get_random_seed())  # Usa una semilla aleatoria única
        num_samples, num_timesteps, num_channels = signal.shape
        signal_interpolated = np.empty((num_samples, num_points, num_channels))

        for i in range(num_samples):
            for j in range(num_channels):
                x_original = np.arange(num_timesteps)
                x_interpolated = np.linspace(0, num_timesteps - 1, num_points)
                interpolation = scipy.interpolate.interp1d(x_original, signal[i, :, j], kind='cubic')
                signal_interpolated[i, :, j] = interpolation(x_interpolated)

        return signal_interpolated
    
    def random_wavelet_transform(self, signal, wavelet='db1'):
        coeffs = pywt.wavedec(signal, wavelet)
        for i in range(1, len(coeffs)):
            coeffs[i] *= np.random.uniform(0.1, 2)  # Randomly scale detail coefficients
        return pywt.waverec(coeffs, wavelet)
    
    def random_phase_shift_transform(self, signal, max_phase_shift=np.pi):
        phase_shift = np.random.uniform(-max_phase_shift, max_phase_shift)
        return np.abs(signal) * np.exp(1j * np.angle(signal) + phase_shift)

    def random_segment_removal_transform(X, min_segments=1, max_segments=None):
        if max_segments is None:
            max_segments = X.shape[0]
        num_segments = np.random.randint(min_segments, min(max_segments, X.shape[0]) + 1)
        segment_starts = np.sort(np.random.choice(X.shape[0], num_segments, replace=False))
        mask = np.ones(X.shape[0], dtype=bool)
        for start in segment_starts:
            mask[start:start+1] = False
        return X[mask]
    
    def random_concatenation_transform(self, signal, min_segments=2, max_segments=5):
        num_segments = np.random.randint(min_segments, max_segments + 1)
        segment_lengths = np.random.randint(1, signal.shape[1], num_segments)
        segment_starts = np.random.randint(0, signal.shape[1] - segment_lengths + 1, num_segments)

        concatenated_signal = np.zeros_like(signal)

        for i in range(signal.shape[0]):
            for j in range(signal.shape[2]):
                for start, length in zip(segment_starts, segment_lengths):
                    concatenated_signal[i, start:start + length, j] = np.random.choice(signal[i, :, j], length)

        return concatenated_signal





    def spectral_distortion_transform(self, X, alpha=0.5):
        """
        Distorts the spectral content of the data.
        """
        # Apply Fourier transform
        X_fft = np.fft.fft(X, axis=1)

        # Apply distortion to the frequency components
        X_distorted_fft = X_fft * np.exp(1j * alpha * np.random.uniform(size=X.shape))

        # Apply inverse Fourier transform
        X_distorted = np.fft.ifft(X_distorted_fft, axis=1).real

        return X_distorted
    
    def phase_modulation_transform(self, X, max_phase_modulation=np.pi/4):
        """
        Modulates the phase of the data.
        """
        phase_shifts = np.random.uniform(-max_phase_modulation, max_phase_modulation, size=X.shape)
        X_modulated = X * np.exp(1j * phase_shifts)

        return X_modulated
    
    def nonlinear_transform(self, X):
        """
        Applies a random nonlinear transformation to the data.
        """
        # Genera una función de transformación aleatoria para cada llamada
        random_func = self.generate_random_nonlinear_func()
        
        # Aplica la función de transformación aleatoria a los datos
        X_transformed = random_func(X)
        
        return X_transformed
    
    def generate_random_nonlinear_func(self):
        """
        Genera una función de transformación no lineal aleatoria.
        """
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        
        # Elije una función aleatoria de un conjunto predefinido
        nonlinear_funcs = [np.sin, np.cos, np.tanh, np.exp]
        random_func = np.random.choice(nonlinear_funcs)
        
        return random_func

    
    
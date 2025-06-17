
import numpy as np
import torch
from scipy.ndimage import rotate

class RandomRotate:
    def __init__(self, random_state, angle_spectrum=10, axes=None, mode='reflect', order=0, **kwargs):
        if axes is None:
            axes = [(1, 0), (2, 1), (2, 0)]
        else:
            assert isinstance(axes, list) and len(axes) > 0
        self.random_state = random_state
        self.angle_spectrum = angle_spectrum
        self.axes = axes
        self.mode = mode
        self.order = order

    def __call__(self, volumes, label, type):
        axis = self.axes[self.random_state.randint(len(self.axes))]
        angle = self.random_state.randint(-self.angle_spectrum, self.angle_spectrum)
        for i in range(len(volumes)+1):
            m = volumes[i] if i != len(volumes) else label
            if m.ndim == 3:
                m = rotate(m, angle, axes=axis, reshape=False, order=self.order, mode=self.mode, cval=-1)
            else:
                channels = [rotate(m[c], angle, axes=axis, reshape=False, order=self.order, mode=self.mode, cval=-1) for c
                            in range(m.shape[0])]
                m = np.stack(channels, axis=0)
            if i != len(volumes):
                volumes[i] = m
            else:
                label = m
        return volumes, label, None

class RandomFlip:
    def __init__(self, random_state, axis_prob=0.5, **kwargs):
        assert random_state is not None, 'RandomState cannot be None'
        self.random_state = random_state
        self.axes = (0, 1, 2)
        self.axis_prob = axis_prob

    def __call__(self, volumes, label, type):

        assert label.ndim in [3, 4], 'Supports only 3D (DxHxW) or 4D (CxDxHxW) images'
        for axis in self.axes:
            if self.random_state.uniform() > self.axis_prob:
                for i in range(len(volumes) + 1):
                    m = volumes[i] if i != len(volumes) else label
                    if m.ndim == 3:
                        m = np.flip(m, axis)
                    else:
                        channels = [np.flip(m[c], axis) for c in range(m.shape[0])]
                        m = np.stack(channels, axis=0)
                    if i != len(volumes):
                        volumes[i] = m
                    else:
                        label = m
        return volumes, label, None

class ThrowFirstZ:
    def __init__(self):
        self.p = None

    def __call__(self, volumes, label, type):
        # volumes[0] = volumes[0][3:,:,:]
        # volumes[1] = volumes[1][3:,:,:]
        # volumes[2] = volumes[2][3:,:,:]
        # volumes[3] = volumes[3][3:,:,:]
        for i in range(len(volumes)):
            volumes[i] = volumes[i][3:, :, :]

        label = label[3:,:,:]
        return volumes, label, None


class RandomCrop:
    def __init__(self, size, **kwargs):
        self.size = size
        self.background = 0
        self.upper_bound1 = 152
        self.upper_bound2 = 240
        self.padding=1

    def __call__(self, volumes, label, type):
        brain_mask = np.zeros_like(volumes[0], dtype=bool)
        for volume in volumes:
            brain_mask |= (volume != volume[0, 0, 0])
        brain = np.where(brain_mask != self.background)

        # 计算裁剪区域的中心点
        center_z = (np.max(brain[0]) + np.min(brain[0])) // 2
        center_y = (np.max(brain[1]) + np.min(brain[1])) // 2
        center_x = (np.max(brain[2]) + np.min(brain[2])) // 2

        # 调用 random_crop 函数来计算裁剪区域
        min_z, max_z = random_crop(center_z - self.size // 2, center_z + self.size // 2, self.size, self.upper_bound1,
                                   type)
        min_y, max_y = random_crop(center_y - self.size // 2, center_y + self.size // 2, self.size, self.upper_bound2,
                                   type)
        min_x, max_x = random_crop(center_x - self.size // 2, center_x + self.size // 2, self.size, self.upper_bound2,
                                   type)

        resizer = (slice(min_z, max_z), slice(min_y, max_y), slice(min_x, max_x))

        for i in range(len(volumes)):
            volumes[i] = volumes[i][resizer]
        label = label[resizer]

        return volumes, label, [min_z, max_z, min_y, max_y, min_x, max_x]

def random_crop(min, max, size, upper_bound, type):
    gap = (max - min)
    if size == gap:
        return min, max
    if gap > size:
        i = (gap-size)//2 if type in ['test', 'val'] else torch.randint(0, gap-size, (1,))[0]
        min = min + i
        max = min + size
    else:
        i = (size-gap)//2 if type in ['test', 'val'] else torch.randint(0, size-gap, (1,))[0]
        if min - i < 0:
            min = 0
            max = min + size
        elif min - i + size >= upper_bound:
            max = upper_bound - 1
            min = max - size
        else:
            min = min - i
            max = min + size
    return min, max

class ToLongTensor:
    def __init__(self):
        self.type = None
    def __call__(self, lbl):

        return torch.from_numpy(lbl).long()


class NpToTensor:
    def __init__(self):
        self.type = None
    def __call__(self, v):
        return torch.tensor(v)


class Standardize:

    def __init__(self):
        self.eps = 1e-10

    def __call__(self, m):
        mean = np.mean(m)
        std = np.std(m)
        return (m - mean) / np.clip(std, a_min=self.eps, a_max=None)

class Normalize:

    def __init__(self):
        self.eps = 1e-10
    def __call__(self, v):
        max_value = np.max(v)
        min_value = np.min(v)
        value_range = max_value - min_value
        norm_0_1 = (v - min_value) / value_range
        return np.clip(2 * norm_0_1 - 1, -1, 1)

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, volumes, label, type):
        crop_size = None
        for t in self.transforms:
            volumes, label, crop_size = t(volumes, label, type)
        return volumes, label, crop_size


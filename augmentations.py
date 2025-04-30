import numpy as np
import torch
import random
import torch.backends.cudnn as cudnn

#x.shape batch_size,num_features,seq_len
def DataTransform(sample, jitter_scale_ratio,max_seg,jitter_ratio):

    weak_aug = scaling(sample, jitter_scale_ratio)
    strong_aug = jitter(permutation(sample, max_segments=max_seg), jitter_ratio)

    return weak_aug.float(), strong_aug.float()


def jitter(x, sigma=0.8):
    # https://arxiv.org/pdf/1706.00527.pdf
    np.random.seed(42)
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)


def scaling(x, sigma=1.1):
    #为batch每个样本上的每个时间步分配一个factor，同一个样本同一个时间步的不同通道间共享一个相同的factor
    #Assign a factor to each <sample,timestep> factor.shape(batch_size,1,seq_len):newaxis for broadcast
    #different channels at the same <sample,timestep> share the same factor
    np.random.seed(42)
    factor = np.random.normal(loc=2., scale=sigma, size=(x.shape[0], x.shape[2]))[:, np.newaxis, :] #batch_size,1,seq_len
    return x * factor


def permutation(x, max_segments=5, seg_mode="random"):
    # x.shape batch_size,num_features,seq_len
    orig_steps = np.arange(x.shape[2])
    np.random.seed(42)
    num_segs = np.random.randint(1, max_segments, size=(x.shape[0]))
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            if seg_mode == "random":
                np.random.seed(42)
                split_points = np.random.choice(x.shape[2] - 2, num_segs[i] - 1, replace=False)
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            else:
                splits = np.array_split(orig_steps, num_segs[i])
            # 随机打乱列表中各个片段的顺序
            random.seed(42)
            random.shuffle(splits)
            # 将打乱后的各个片段展平成一个列表
            warp = [elem for segment in splits for elem in segment]
            ret[i] = pat[:,warp]#pat[0,warp]
        else:
            ret[i] = pat
    return torch.from_numpy(ret)


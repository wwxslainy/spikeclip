import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

# 从 JSON 文件加载配置
def load_config(json_file):
    with open(json_file, 'r') as f:
        config = json.load(f)
    return config


def generate_json_from_folder(base_path, output_json_path):
    """
    遍历给定路径下的所有子文件夹，为每个文件夹生成路径和对应的标签（文件夹名称）。
    
    :param base_path: 数据集的根目录
    :param output_json_path: 生成的 JSON 文件路径
    """
    folder_label_mapping = {}

    # 遍历给定路径下的所有文件夹
    for root, dirs, files in os.walk(base_path):
        # 排除不包含子文件夹的路径
        if not dirs:
            # 生成文件夹路径和标签
            label = os.path.basename(os.path.dirname(root))  # 获取父文件夹作为标签
            folder_label_mapping[root] = label

    # 将字典保存为 JSON 文件
    with open(output_json_path, 'w', encoding='utf-8') as json_file:
        json.dump(folder_label_mapping, json_file, ensure_ascii=False, indent=4)

    print(f"JSON 文件已保存到 {output_json_path}")


def get_spike_matrix(filename, spike_height, spike_width, flipud=False):
    """
    从 .dat 文件读取脉冲数据并返回脉冲矩阵。
    """
    with open(filename, 'rb') as file:
        video_seq = file.read()

    video_seq = np.frombuffer(video_seq, 'b')
    video_seq = np.array(video_seq).astype(np.byte)

    img_size = spike_height * spike_width
    total_data_length = len(video_seq)
    
    # 计算帧数（通道数）
    channels = total_data_length // (img_size // 8)
    
    SpikeMatrix = np.zeros([channels, spike_height, spike_width], np.byte)

    pix_id = np.arange(0, spike_height * spike_width)
    pix_id = np.reshape(pix_id, (spike_height, spike_width))
    comparator = np.left_shift(1, np.mod(pix_id, 8))
    byte_id = pix_id // 8

    for channel_id in range(channels):
        id_start = channel_id * img_size // 8
        id_end = id_start + img_size // 8
        cur_info = video_seq[id_start:id_end]
        data = cur_info[byte_id]
        result = np.bitwise_and(data, comparator)

        if flipud:
            SpikeMatrix[channel_id, :, :] = np.flipud((result == comparator))
        else:
            SpikeMatrix[channel_id, :, :] = (result == comparator)

    return SpikeMatrix


class SpikingVideoDataset(Dataset):
    def __init__(self, json_file, spike_h, spike_w, device, target_frames=25, channels=10):
        self.device = device
        self.target_frames = target_frames  # 目标帧数
        self.channels = channels  # 每帧保留的通道数
        config = load_config(json_file)
        self.dat_folder_paths = list(config.keys())  # 获取文件夹路径
        self.labels = list(config.values())  # 获取对应的标签
        self.spike_h = spike_h
        self.spike_w = spike_w

    def __len__(self):
        return len(self.dat_folder_paths)

    def __getitem__(self, idx):
        folder_path = self.dat_folder_paths[idx]  # 获取当前视频文件夹路径
        caption = self.labels[idx]  # 获取对应的标签

        # 获取所有 .dat 文件
        dat_files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.dat')])
        
        video_frames = []

        # 读取每个 .dat 文件并转换为脉冲矩阵
        for dat_file in dat_files:
            frames = get_spike_matrix(dat_file, self.spike_h, self.spike_w)
            # 确保每个脉冲矩阵的通道数一致
            if frames.shape[0] > self.channels:
                frames = frames[:self.channels, :, :]  # 截断到目标通道数
            elif frames.shape[0] < self.channels:
                padding = np.zeros((self.channels - frames.shape[0], self.spike_h, self.spike_w), dtype=frames.dtype)
                frames = np.concatenate([frames, padding], axis=0)  # 填充到目标通道数
            video_frames.append(frames)

        # 将所有帧堆叠成一个 4D 数组，形状为 [T, C, H, W]
        video_frames = np.stack(video_frames, axis=0)  # T: 代表时间维度（帧数）

        # 调整帧数到目标帧数
        video_frames = self.select_fixed_interval_frames(video_frames, self.target_frames)

        # 保留每帧的前10个通道
        video_frames = video_frames[:, :self.channels, :, :]  # 形状变为 (target_frames, 10, H, W)

        # 转换为 4D tensor [T, C, H, W]
        video_frames = torch.tensor(video_frames).float()  # 不直接转移到 GPU
        return video_frames, caption

    @staticmethod
    def select_fixed_interval_frames(video_frames, target_frames):
        current_frames = video_frames.shape[0]
        if current_frames > target_frames:
            interval = current_frames // target_frames
            indices = np.arange(0, current_frames, interval)[:target_frames]
            return video_frames[indices]
        elif current_frames < target_frames:
            pad_frames = target_frames - current_frames
            pad = np.zeros((pad_frames, *video_frames.shape[1:]), dtype=video_frames.dtype)
            return np.concatenate([video_frames, pad], axis=0)
        else:
            return video_frames
    
def create_data_loaders(json_file, spike_h, spike_w, batch_size, val_split=0.2, device='cpu'):
    # 创建数据集
    dataset = SpikingVideoDataset(json_file=json_file, spike_h=spike_h, spike_w=spike_w, device=device, target_frames=25, channels=10)

    # 计算训练集和验证集的分割点
    total_samples = len(dataset)
    val_size = int(total_samples * val_split)
    train_size = total_samples - val_size

    # 使用 random_split 切分数据集
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


# 使用例子
if __name__ == "__main__":
    # 假设有一个 JSON 文件，保存了文件夹路径和标签
    json_file_path = "./UCF101_3_10_.json"
    
    train_loader, val_loader = create_data_loaders(
        json_file=json_file_path, spike_h=240, spike_w=320, batch_size=1, device='cuda'
    )
    
    # 查看训练集中的一个批次
    for frames, caption in train_loader:
        print("Frames shape:", frames.shape)  # 期待输出 [B, T, C, H, W]
        print("Caption:", caption)
        break
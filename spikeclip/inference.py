import os
import torch
import numpy as np
from clip.model import CLIP  # 确保你的 CLIP 模型已正确导入
from dataset import get_spike_matrix  # 确保你有处理 .dat 文件的函数
from torchvision import transforms

# 配置
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = "/root/a_new_clip/weight/try_2batch.pth"
SPIKE_H, SPIKE_W = 240, 320  # 图像高度和宽度
TARGET_FRAMES = 224  # 固定帧数
LABELS = ["playbasketball", "playingpiano", "skiing"]  # 替换为你的分类标签

# 数据预处理函数
def preprocess_video(video_folder, spike_h, spike_w, target_frames):
    """
    读取视频文件夹中的 .dat 文件，将其转换为脉冲矩阵，并调整帧数。

    :param video_folder: 视频文件夹路径
    :param spike_h: 每帧的高度
    :param spike_w: 每帧的宽度
    :param target_frames: 固定帧数
    :return: 预处理后的视频张量，形状为 [target_frames, C, H, W]
    """
    dat_files = sorted([os.path.join(video_folder, f) for f in os.listdir(video_folder) if f.endswith('.dat')])
    video_frames = []

    for dat_file in dat_files:
        frames = get_spike_matrix(dat_file, spike_h, spike_w)  # 自定义函数，读取 .dat 文件为矩阵
        video_frames.append(frames)

    # 将所有帧堆叠成 4D 数组 [T, C, H, W]
    video_frames = np.stack(video_frames, axis=0)

    # 均匀采样调整帧数
    if video_frames.shape[0] > target_frames:
        indices = np.linspace(0, video_frames.shape[0] - 1, target_frames, dtype=int)
        video_frames = video_frames[indices]
    elif video_frames.shape[0] < target_frames:
        pad_frames = target_frames - video_frames.shape[0]
        pad = np.zeros((pad_frames, *video_frames.shape[1:]), dtype=video_frames.dtype)
        video_frames = np.concatenate([video_frames, pad], axis=0)

    # 转换为 Tensor 并返回
    video_tensor = torch.tensor(video_frames).float().to(DEVICE)
    return video_tensor

# 初始化模型
def load_model(model_path, device):
    """
    加载模型及权重。

    :param model_path: 模型权重文件路径
    :param device: 设备（CPU 或 GPU）
    :return: 加载后的模型
    """
    model = CLIP(
        embed_dim=64,
        image_resolution=(SPIKE_H, SPIKE_W),
        vision_layers=(2, 2, 2, 2),
        vision_width=64,
        context_length=77,
        vocab_size=49408,
        transformer_width=64,
        transformer_heads=2,
        transformer_layers=2,
        input_channels=50
    ).to(device)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()  # 切换到推理模式
        print(f"Loaded model weights from {model_path}")
    else:
        raise FileNotFoundError(f"No model weights found at {model_path}")

    return model

# 推理函数
def predict(video_folder, model, labels):
    """
    对给定的视频文件夹进行推理，输出分类结果。

    :param video_folder: 视频文件夹路径
    :param model: 加载的模型
    :param labels: 分类标签列表
    :return: 分类结果
    """
    # 预处理视频
    video_tensor = preprocess_video(video_folder, SPIKE_H, SPIKE_W, TARGET_FRAMES)
    video_tensor = video_tensor.unsqueeze(0)  # 增加 batch 维度，形状为 [1, T, C, H, W]

    # 创建一个伪文本输入（长度为 77，填充全零）
    dummy_text = torch.zeros((1, 77), dtype=torch.long).to(DEVICE)

    # 推理
    with torch.no_grad():
        logits_per_image, _ = model(video_tensor, dummy_text)  # 使用伪文本输入
        pred_idx = logits_per_image.argmax(dim=1).item()  # 获取预测类别索引
        pred_label = labels[pred_idx]  # 获取预测类别名称

    return pred_label

# 主函数
if __name__ == "__main__":
    video_path = "/root/train_s/skiing/v_Skiing_g01_c05"  # 替换为你的视频文件夹路径
    model = load_model(MODEL_PATH, DEVICE)  # 加载模型
    result = predict(video_path, model, LABELS)  # 推理并获取分类结果
    print(f"The predicted label for the video is: {result}")

import os
import torch
import numpy as np
from clip.simple_tokenizer import SimpleTokenizer  # 确保你已定义这个类
from clip.model import CLIP

# 配置设备和路径
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = "./weight/try_2_17.pth"
DAT_FOLDER_PATH = "D:/Ccc-lab-robot/a_new_clip/video_s/diving/v_Diving_g01_c02"  # 替换为你的 .dat 文件所在的文件夹路径
SPIKE_H, SPIKE_W = 240, 320  # 图像高度和宽度
TARGET_FRAMES = 25  # 目标帧数
CHANNELS = 10  # 每帧保留的通道数

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

def load_model():
    model = CLIP(
        embed_dim=256,
        image_resolution=(SPIKE_H, SPIKE_W),
        vision_layers=(2, 2, 4, 2),
        vision_width=256,
        context_length=77,
        vocab_size=49408,
        transformer_width=128,
        transformer_heads=4,
        transformer_layers=4,
        input_channels=64
    ).to(DEVICE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH))
        print(f"Loaded model weights from {MODEL_PATH}")
    else:
        raise FileNotFoundError(f"No pre-trained model found at {MODEL_PATH}")
    model.eval()
    return model

def encode_texts(tokenizer, captions):
    tokenized_texts = []
    for caption in captions:
        tokenized = tokenizer.encode(caption)
        tokenized = tokenized[:77]
        tokenized += [0] * (77 - len(tokenized))
        tokenized_texts.append(tokenized)
    return torch.tensor(tokenized_texts).to(DEVICE)

def preprocess_dat_files(dat_folder_path, spike_h, spike_w, target_frames, channels):
    dat_files = sorted([os.path.join(dat_folder_path, f) for f in os.listdir(dat_folder_path) if f.endswith('.dat')])
    
    video_frames = []

    # 读取每个 .dat 文件并转换为脉冲矩阵
    for dat_file in dat_files:
        frames = get_spike_matrix(dat_file, spike_h, spike_w)
        if frames.shape[0] > channels:
            frames = frames[:channels, :, :]
        elif frames.shape[0] < channels:
            padding = np.zeros((channels - frames.shape[0], spike_h, spike_w), dtype=frames.dtype)
            frames = np.concatenate([frames, padding], axis=0)
        video_frames.append(frames)

    # 将所有帧堆叠成一个 4D 数组，形状为 [T, C, H, W]
    video_frames = np.stack(video_frames, axis=0)

    # 调整帧数到目标帧数
    current_frames = video_frames.shape[0]
    if current_frames > target_frames:
        interval = current_frames // target_frames
        indices = np.arange(0, current_frames, interval)[:target_frames]
        video_frames = video_frames[indices]
    elif current_frames < target_frames:
        pad_frames = target_frames - current_frames
        pad = np.zeros((pad_frames, *video_frames.shape[1:]), dtype=video_frames.dtype)
        video_frames = np.concatenate([video_frames, pad], axis=0)

    # 保留每帧的前10个通道
    video_frames = video_frames[:, :channels, :, :]

    # 转换为 4D tensor [T, C, H, W]
    video_frames = torch.tensor(video_frames).float().unsqueeze(0)  # 添加 batch 维度
    return video_frames

def test_single_video(model, video_tensor, tokenizer):
    with torch.no_grad():
        video_tensor = video_tensor.to(DEVICE)
        
        # 假设我们有一些预定义的文本标签
        predefined_captions = ["ApplyLipstick", "Archery", "BabyCrawling","BalanceBeam","BaseballPitch","Basketball","BasketballDunk","BenchPress","Biking"] # 根据实际情况修改
        tokenized_texts = encode_texts(tokenizer, predefined_captions)

        # 对每个预定义标签分别进行前向传播
        similarities = []
        for predefined_caption in predefined_captions:
            tokenized_text = encode_texts(tokenizer, [predefined_caption])
            logits_per_image, logits_per_text = model(video_tensor, tokenized_text)

            # 计算相似度
            similarity = logits_per_image @ logits_per_text.t()
            similarities.append(similarity.item())

        # 打印相似度得分
        print(f"Similarity scores: {similarities}")

        # 找到最相似的配对
        predicted_caption_idx = similarities.index(max(similarities))
        predicted_caption = predefined_captions[predicted_caption_idx]

        return predicted_caption

if __name__ == "__main__":
    # 加载模型
    model = load_model()
    tokenizer = SimpleTokenizer()

    # 预处理 .dat 文件
    video_tensor = preprocess_dat_files(DAT_FOLDER_PATH, SPIKE_H, SPIKE_W, TARGET_FRAMES, CHANNELS)

    # 进行推理并输出预测标签
    predicted_label = test_single_video(model, video_tensor, tokenizer)
    print(f"Predicted Label: {predicted_label}")
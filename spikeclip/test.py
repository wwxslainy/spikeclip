import torch
from dataset import SpikingVideoDataset
from clip.simple_tokenizer import SimpleTokenizer
from torch.utils.data import DataLoader
from clip.model import CLIP
import os

# 配置设备和路径
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = "./weight/try_2_17.pth"
JSON_TEST_CONFIG_PATH = "./UCF101_3_10_.json"
SPIKE_H, SPIKE_W = 240, 320
BATCH_SIZE = 8  # 增大batch size
PREDEFINED_CAPTIONS = ["ApplyLipstick", "Archery", "BabyCrawling","BalanceBeam","BaseballPitch","Basketball","BasketballDunk","BenchPress","Biking"]

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

def create_test_loader():
    test_dataset = SpikingVideoDataset(JSON_TEST_CONFIG_PATH, SPIKE_H, SPIKE_W, DEVICE)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return test_dataloader

def encode_texts(tokenizer, captions):
    tokenized_texts = []
    for caption in captions:
        tokenized = tokenizer.encode(caption)
        tokenized = tokenized[:77]
        tokenized += [0] * (77 - len(tokenized))
        tokenized_texts.append(tokenized)
    return torch.tensor(tokenized_texts).to(DEVICE)

def test_model(model, test_dataloader):
    tokenizer = SimpleTokenizer()
    
    # 预编码所有文本标签
    text_tokens = encode_texts(tokenizer, PREDEFINED_CAPTIONS)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)  # 使用文本编码器
    
    total_correct = 0
    all_predictions = []
    
    with torch.no_grad():
        for batch_idx, (spike_matrices, captions) in enumerate(test_dataloader):
            spike_matrices = spike_matrices.to(DEVICE)
            
            # 批量编码图像特征
            image_features = model.encode_image(spike_matrices)
            
            # 标准化特征
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # 计算相似度矩阵 [batch_size, num_classes]
            similarity = (image_features @ text_features_norm.T) * model.logit_scale.exp()
            
            # 获取预测结果
            pred_indices = similarity.argmax(dim=1)
            predictions = [PREDEFINED_CAPTIONS[idx] for idx in pred_indices.cpu().numpy()]
            
            # 统计正确率
            for pred, true in zip(predictions, captions):
                all_predictions.append({"predicted": pred, "true": true})
                if pred == true:
                    total_correct += 1
    
    # 输出总体准确率
    total_samples = len(all_predictions)
    accuracy = total_correct / total_samples * 100
    print(f"\nTest Accuracy: {accuracy:.2f}% ({total_correct}/{total_samples})")
    
    # 输出每个样本的预测结果
    print("\nDetailed Predictions:")
    for idx, pred in enumerate(all_predictions):
        status = "✓" if pred["predicted"] == pred["true"] else "✗"
        print(f"Sample {idx+1}: {status}")
        print(f"  Predicted: {pred['predicted']}")
        print(f"  True Label: {pred['true']}\n")

if __name__ == "__main__":
    model = load_model()
    test_dataloader = create_test_loader()
    test_model(model, test_dataloader)
import os 
import torch 
import matplotlib.pyplot as plt 
from torch.utils.data import DataLoader, random_split 
from dataset import SpikingVideoDataset 
from clip.simple_tokenizer import SimpleTokenizer # 确保你已定义这两个类 
from clip.model import CLIP 
import gc # 用于垃圾回收

# 配置设备和路径 
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' 
MODEL_PATH = "./weight/try_2_20.pth" 
JSON_CONFIG_PATH = "./UCF101_3_10_.json" 
SPIKE_H, SPIKE_W = 240, 320 # 图像高度和宽度 
BATCH_SIZE = 4 
TOTAL_EPOCHS = 50 
VAL_SPLIT = 0.2 
LEARNING_RATE = 1e-5 
torch.cuda.empty_cache() 

# 初始化模型 
def initialize_model(): 
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
        print(f"No pre-trained model found at {MODEL_PATH}, starting from scratch.")
    return model

# 创建数据加载器 
def create_data_loaders(json_file=JSON_CONFIG_PATH, spike_h=SPIKE_H, spike_w=SPIKE_W, batch_size=BATCH_SIZE, val_split=0.2, device='cpu', num_workers=1):
    dataset = SpikingVideoDataset(
        json_file=json_file,
        spike_h=spike_h,
        spike_w=spike_w,
        device=device,
        target_frames=25
    )
    total_samples = len(dataset)
    val_size = int(total_samples * val_split)
    train_size = total_samples - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device != 'cpu')
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device != 'cpu')
    )
    return train_loader, val_loader

# 文本编码 
def encode_texts(tokenizer, captions):
    tokenized_texts = []
    for caption in captions:
        tokenized = tokenizer.encode(caption)
        tokenized = tokenized[:77]
        tokenized += [0] * (77 - len(tokenized))
        tokenized_texts.append(tokenized)
    return torch.tensor(tokenized_texts).to(DEVICE)

# 绘制准确率曲线 
def plot_accuracies(train_acc_i, train_acc_t, val_acc_i, val_acc_t):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, TOTAL_EPOCHS + 1), train_acc_i, label="Train I->T Accuracy")
    plt.plot(range(1, TOTAL_EPOCHS + 1), train_acc_t, label="Train T->I Accuracy")
    plt.plot(range(1, TOTAL_EPOCHS + 1), val_acc_i, label="Val I->T Accuracy")
    plt.plot(range(1, TOTAL_EPOCHS + 1), val_acc_t, label="Val T->I Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.savefig("./weight/accuracy_plot1.png")
    print("Accuracy plot saved!")

def train_and_validate(model, train_dataloader, val_dataloader):
    tokenizer = SimpleTokenizer()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_acc_i_list, train_acc_t_list = [], []
    val_acc_i_list, val_acc_t_list = [], []

    for epoch in range(TOTAL_EPOCHS):
        # 训练阶段
        model.train()
        total_loss, total_correct_i, total_correct_t, total_samples = 0, 0, 0, 0
        for spike_matrices, captions in train_dataloader:
            spike_matrices = spike_matrices.to(DEVICE)
            tokenized_texts = encode_texts(tokenizer, captions)
            logits_per_image, logits_per_text = model(spike_matrices, tokenized_texts)
            targets = torch.arange(logits_per_image.size(0)).to(DEVICE)
            loss_i = torch.nn.functional.cross_entropy(logits_per_image, targets)
            loss_t = torch.nn.functional.cross_entropy(logits_per_text, targets)
            loss = (loss_i + loss_t) / 2
            total_loss += loss.item()
            pred_i = logits_per_image.argmax(dim=1)
            pred_t = logits_per_text.argmax(dim=1)
            correct_i = (pred_i == targets).sum().item()
            correct_t = (pred_t == targets).sum().item()
            total_correct_i += correct_i
            total_correct_t += correct_t
            total_samples += len(targets)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            del spike_matrices, tokenized_texts, logits_per_image, logits_per_text, targets, loss_i, loss_t, loss
            torch.cuda.empty_cache()

        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}/{TOTAL_EPOCHS} | Train Loss: {avg_train_loss:.4f}")

        # 验证阶段
        model.eval()
        total_val_loss, total_correct_val_i, total_correct_val_t, total_samples_val = 0, 0, 0, 0
        with torch.no_grad():
            for spike_matrices, captions in val_dataloader:
                spike_matrices = spike_matrices.to(DEVICE)
                tokenized_texts = encode_texts(tokenizer, captions)
                logits_per_image, logits_per_text = model(spike_matrices, tokenized_texts)
                targets = torch.arange(logits_per_image.size(0)).to(DEVICE)
                loss_i = torch.nn.functional.cross_entropy(logits_per_image, targets)
                loss_t = torch.nn.functional.cross_entropy(logits_per_text, targets)
                loss = (loss_i + loss_t) / 2
                total_val_loss += loss.item()
                pred_i = logits_per_image.argmax(dim=1)
                pred_t = logits_per_text.argmax(dim=1)
                correct_i = (pred_i == targets).sum().item()
                correct_t = (pred_t == targets).sum().item()
                total_correct_val_i += correct_i
                total_correct_val_t += correct_t
                total_samples_val += len(targets)
                del spike_matrices, tokenized_texts, logits_per_image, logits_per_text, targets, pred_i, pred_t
                torch.cuda.empty_cache()

        avg_val_loss = total_val_loss / len(val_dataloader)
        print(f"Epoch {epoch+1}/{TOTAL_EPOCHS} | Val Loss: {avg_val_loss:.4f}")

        # 每5个epoch或最后一个epoch计算准确率
        if (epoch + 1) % 5 == 0 or epoch == TOTAL_EPOCHS - 1:
            train_acc_i = total_correct_i / total_samples * 100
            train_acc_t = total_correct_t / total_samples * 100
            val_acc_i = total_correct_val_i / total_samples_val * 100
            val_acc_t = total_correct_val_t / total_samples_val * 100
            train_acc_i_list.append(train_acc_i)
            train_acc_t_list.append(train_acc_t)
            val_acc_i_list.append(val_acc_i)
            val_acc_t_list.append(val_acc_t)
            print(f"Epoch {epoch+1}/{TOTAL_EPOCHS} | "
                  f"Train Acc: I->T {train_acc_i:.2f}% T->I {train_acc_t:.2f}% | "
                  f"Val Acc: I->T {val_acc_i:.2f}% T->I {val_acc_t:.2f}%")

        # 定期保存模型
        torch.save(model.state_dict(), MODEL_PATH)
        gc.collect()
        torch.cuda.empty_cache()

    # 绘制准确率曲线
    plot_accuracies(train_acc_i_list, train_acc_t_list, val_acc_i_list, val_acc_t_list)

# 主函数 
if __name__ == "__main__": 
    model = initialize_model()
    train_dataloader, val_dataloader = create_data_loaders()
    train_and_validate(model, train_dataloader, val_dataloader)
import torch

def custom_collate_fn(batch):
    """
    自定义 collate_fn 函数，用于处理 DataLoader 返回的数据。
    :param batch: 数据批次，每个元素是一个 (spike_matrix, caption) 元组
    :return: 处理后的数据批次
    """
    spike_matrices, captions = zip(*batch)
    
    # 如果 spike_matrices 中有任何张量需要梯度计算，调用 .detach() 来分离它们
    spike_matrices = [s.detach() if s.requires_grad else s for s in spike_matrices]
    
    # 将 spike_matrices 转换为张量并堆叠
    spike_matrices = torch.stack(spike_matrices, dim=0)
    
    # 将 captions 转换为张量或保持原样（取决于你的需求）
    if isinstance(captions[0], torch.Tensor):
        captions = torch.stack(captions, dim=0)
    else:
        captions = list(captions)
    
    return spike_matrices, captions
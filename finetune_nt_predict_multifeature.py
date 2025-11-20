#!/usr/bin/env python3
# finetune_nt_predict_multifeature.py
"""
多特征版本的 Nucleotide Transformer 预测脚本
支持序列特征 + 数值特征的联合预测
"""

import argparse
import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, AutoConfig
from tqdm.auto import tqdm
import numpy as np

# ---------------------
# 模型定义（与训练时相同）
# ---------------------
class MultiFeatureNTClassificationModel(nn.Module):
    def __init__(self, backbone, num_numerical_features, num_labels=2, dropout=0.3):
        super().__init__()
        self.backbone = backbone
        
        # 获取隐藏层维度
        if hasattr(backbone, 'config') and hasattr(backbone.config, 'hidden_size'):
            hidden_size = backbone.config.hidden_size
        elif hasattr(backbone, 'config') and hasattr(backbone.config, 'd_model'):
            hidden_size = backbone.config.d_model
        else:
            try:
                sample_param = next(backbone.parameters())
                hidden_size = sample_param.shape[0]
            except StopIteration:
                hidden_size = 1024
        self.hidden_size = hidden_size
        
        # 数值特征处理网络
        self.numerical_processor = nn.Sequential(
            nn.Linear(num_numerical_features, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 融合分类器（序列特征 + 数值特征）
        combined_dim = hidden_size + 64
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(combined_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_labels)
        )

    def forward(self, input_ids=None, attention_mask=None, numerical_features=None, **kwargs):
        # 处理序列特征
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        
        if hasattr(outputs, 'last_hidden_state'):
            last_hidden = outputs.last_hidden_state
        elif isinstance(outputs, (tuple, list)) and len(outputs) > 0:
            last_hidden = outputs[0]
        else:
            raise RuntimeError("无法获取 backbone 的隐藏状态")
        
        # 序列特征池化（与训练时相同）
        avg_pool = last_hidden.mean(dim=1)
        max_pool = last_hidden.max(dim=1)[0]
        seq_features = avg_pool + max_pool
        
        # 处理数值特征
        if numerical_features is not None:
            numerical_features = self.numerical_processor(numerical_features)
            # 融合特征
            combined_features = torch.cat([seq_features, numerical_features], dim=1)
        else:
            combined_features = seq_features
        
        logits = self.classifier(combined_features)
        return logits

# ---------------------
# 数据处理（与训练时相同）
# ---------------------
class MultiFeatureSeqDataset(torch.utils.data.Dataset):
    def __init__(self, sequences, features, tokenizer, max_length=512, feature_dim=6):
        self.seqs = sequences
        self.features = features
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.feature_dim = feature_dim

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        feature_vec = self.features[idx]
        
        # Tokenize 序列
        enc = self.tokenizer(
            seq,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["numerical_features"] = torch.tensor(feature_vec, dtype=torch.float)
        
        return item

def read_csv_multifeature(path: str):
    """读取CSV文件，返回序列和数值特征列表"""
    seqs, features = [], []
    
    try:
        df = pd.read_csv(path)
        
        # 检查必要的列
        if 'Off' not in df.columns:
            raise ValueError("CSV文件必须包含 'Off' 列")
        
        # 序列列
        seqs = df['Off'].astype(str).tolist()
        
        # 特征列
        feature_columns = ['Epi_satics', 'CFD_score', 'CCTop_Score', 'Moreno_Score', 'CROPIT_Score', 'MIT_Score']
        available_columns = []
        for col in feature_columns:
            if col in df.columns:
                available_columns.append(col)
            else:
                print(f"警告: 特征列 {col} 不存在，跳过")
        
        # 提取数值特征
        if available_columns:
            features = df[available_columns].values.tolist()
        else:
            features = [[0] * 6 for _ in range(len(seqs))]  # 默认特征
        
        print(f"使用的特征列: {available_columns}")
        print(f"特征维度: {len(available_columns)}")
        
    except Exception as e:
        print(f"读取CSV文件失败: {e}")
        raise
    
    return seqs, features, df

# ---------------------
# 本地模型加载函数
# ---------------------
def load_model_and_tokenizer_locally(model_path, device):
    """从本地目录加载模型和tokenizer"""
    print("🔧 正在从本地加载模型和tokenizer...")
    
    # 检查是否是HuggingFace模型目录
    if os.path.isdir(model_path):
        model_dir = model_path
    else:
        checkpoint_dir = os.path.dirname(model_path)
        possible_dirs = [
            os.path.join(checkpoint_dir, "model"),
            checkpoint_dir,
            os.path.dirname(checkpoint_dir)
        ]
        
        model_dir = None
        for dir_path in possible_dirs:
            if os.path.exists(os.path.join(dir_path, "config.json")):
                model_dir = dir_path
                break
        
        if model_dir is None:
            print("❌ 未找到本地模型文件")
            return None, None
    
    # 加载tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            local_files_only=True,
            trust_remote_code=True
        )
        print("✅ Tokenizer本地加载成功")
    except Exception as e:
        print(f"❌ Tokenizer本地加载失败: {e}")
        return None, None
    
    # 加载模型配置与backbone
    try:
        config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
        backbone = AutoModel.from_pretrained(
            model_dir,
            config=config,
            local_files_only=True,
            trust_remote_code=True
        )
        print("✅ 模型本地加载成功")
    except Exception as e:
        print(f"❌ 模型本地加载失败: {e}")
        return None, None
    
    return backbone, tokenizer

def download_model_locally(model_name="InstaDeepAI/nucleotide-transformer-500m-1000g", local_dir="./local_models"):
    """下载模型到本地目录"""
    print(f"📥 正在下载模型 {model_name} 到本地目录 {local_dir}...")
    
    os.makedirs(local_dir, exist_ok=True)
    model_dir = os.path.join(local_dir, model_name.split('/')[-1])
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=local_dir
        )
        tokenizer.save_pretrained(model_dir)
        print("✅ Tokenizer下载并保存成功")
        
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=local_dir
        )
        model.save_pretrained(model_dir)
        print("✅ 模型下载并保存成功")
        
        print(f"📁 模型已保存到: {model_dir}")
        return model_dir
        
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return None

# ---------------------
# 模型加载
# ---------------------
def load_trained_model(checkpoint_path, local_model_dir, device, feature_dim=6):
    """加载训练好的多特征模型"""
    print(f"加载模型检查点: {checkpoint_path}")
    
    # 加载backbone（本地）
    backbone, tokenizer = load_model_and_tokenizer_locally(local_model_dir, device)
    if backbone is None:
        return None, None
    
    # 创建多特征模型架构
    model = MultiFeatureNTClassificationModel(
        backbone, 
        num_numerical_features=feature_dim, 
        num_labels=2
    )
    
    # 加载训练好的权重
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        print("✅ 模型权重加载成功")
        
        # 检查特征维度是否匹配
        saved_feature_dim = checkpoint.get("feature_dim", feature_dim)
        if saved_feature_dim != feature_dim:
            print(f"⚠️ 警告: 保存的特征维度({saved_feature_dim})与当前特征维度({feature_dim})不匹配")
            
    except Exception as e:
        print(f"❌ 权重加载失败: {e}")
        return None, None

    # 确保模型各部分dtype一致
    backbone_dtype = None
    for p in model.backbone.parameters():
        backbone_dtype = p.dtype
        break
    if backbone_dtype is None:
        backbone_dtype = torch.float32

    try:
        model.numerical_processor.to(dtype=backbone_dtype)
        model.classifier.to(dtype=backbone_dtype)
    except Exception as e:
        print(f"⚠️ 转换dtype时遇到问题: {e}")

    # 将整个模型移动到设备
    try:
        model.to(device)
    except Exception as e:
        print(f"❌ 将模型移动到设备 {device} 失败: {e}")
        return None, None

    model.eval()
    return model, tokenizer

# ---------------------
# 预测函数
# ---------------------
def predict(model, dataloader, device):
    """进行多特征预测"""
    model.eval()
    all_probs = []
    all_logits = []
    all_predictions = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="预测中"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            numerical_features = batch["numerical_features"].to(device)
            
            logits = model(
                input_ids=input_ids, 
                attention_mask=attention_mask,
                numerical_features=numerical_features
            )
            probs = torch.softmax(logits, dim=-1)
            
            all_probs.extend(probs.cpu().numpy())
            all_logits.extend(logits.cpu().numpy())
            all_predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())
    
    return all_probs, all_logits, all_predictions

# ---------------------
# 主函数
# ---------------------
def main():
    parser = argparse.ArgumentParser(description="多特征 Nucleotide Transformer 预测脚本")
    parser.add_argument("--checkpoint", type=str, required=True, 
                       help="训练好的模型检查点路径")
    parser.add_argument("--input_csv", type=str, required=True,
                       help="输入数据CSV文件路径")
    parser.add_argument("--output_csv", type=str, required=True,
                       help="预测结果输出CSV文件路径")
    parser.add_argument("--local_model_dir", type=str, 
                       default="./local_models/nucleotide-transformer-500m-1000g",
                       help="本地模型目录路径")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="预测批次大小")
    parser.add_argument("--max_length", type=int, default=512,
                       help="最大序列长度")
    parser.add_argument("--device", type=str, default="cuda",
                       help="推理设备 (cuda/cpu)")
    parser.add_argument("--download_model", action="store_true",
                       help="如果本地没有模型，先下载模型")
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device(args.device if (torch.cuda.is_available() and 'cuda' in args.device) else "cpu")
    print(f"使用设备: {device}")
    
    # 检查文件是否存在
    if not os.path.exists(args.checkpoint):
        print(f"❌ 检查点文件不存在: {args.checkpoint}")
        return
    
    if not os.path.exists(args.input_csv):
        print(f"❌ 输入文件不存在: {args.input_csv}")
        return
    
    # 检查本地模型是否存在
    if not os.path.exists(args.local_model_dir) and args.download_model:
        print("本地模型目录不存在，开始下载...")
        model_dir = download_model_locally(local_dir=os.path.dirname(args.local_model_dir))
        if model_dir is None:
            return
    elif not os.path.exists(args.local_model_dir):
        print(f"❌ 本地模型目录不存在: {args.local_model_dir}")
        print("请使用 --download_model 参数自动下载")
        return
    
    # 加载数据并获取特征维度
    print("正在加载数据...")
    sequences, features, original_df = read_csv_multifeature(args.input_csv)
    feature_dim = len(features[0]) if features else 6
    print(f"加载了 {len(sequences)} 条序列，特征维度: {feature_dim}")
    
    # 加载模型和tokenizer
    model, tokenizer = load_trained_model(args.checkpoint, args.local_model_dir, device, feature_dim)
    if model is None:
        print("❌ 模型加载失败")
        return
    
    # 创建数据集和数据加载器
    dataset = MultiFeatureSeqDataset(sequences, features, tokenizer, args.max_length, feature_dim)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    # 进行预测
    print("开始多特征预测...")
    probabilities, logits, predictions = predict(model, dataloader, device)
    
    # 准备输出结果
    results_df = original_df.copy()
    
    # 添加预测结果
    results_df['prediction'] = predictions
    results_df['probability_class_0'] = [prob[0] for prob in probabilities]
    results_df['probability_class_1'] = [prob[1] for prob in probabilities]
    results_df['confidence'] = np.max(probabilities, axis=1)
    
    # 添加预测标签
    results_df['predicted_label'] = results_df['prediction'].map({0: 'negative', 1: 'positive'})
    
    # 保存结果
    results_df.to_csv(args.output_csv, index=False)
    print(f"✅ 预测完成！结果已保存到: {args.output_csv}")
    
    # 打印统计信息
    print("\n📊 预测统计:")
    print(f"   总样本数: {len(results_df)}")
    print(f"   预测为正类的样本数: {sum(predictions)}")
    print(f"   预测为负类的样本数: {len(predictions) - sum(predictions)}")
    print(f"   平均置信度: {np.mean(results_df['confidence']):.4f}")

if __name__ == "__main__":
    main()
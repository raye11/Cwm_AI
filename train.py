import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
import numpy as np
import json
from tqdm import tqdm
import os
import requests

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

class CommunityDataset(Dataset):
    def __init__(self, texts, category_labels, emotion_labels, urgency_labels, tokenizer, max_length=128):
        self.texts = texts
        self.category_labels = category_labels
        self.emotion_labels = emotion_labels
        self.urgency_labels = urgency_labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'category_labels': torch.tensor(self.category_labels[idx], dtype=torch.long),
            'emotion_labels': torch.tensor(self.emotion_labels[idx], dtype=torch.long),
            'urgency_labels': torch.tensor(self.urgency_labels[idx], dtype=torch.long)
        }

class MultiTaskCommunityModel(nn.Module):
    def __init__(self, model_name, num_categories, num_emotions, num_urgency):
        super(MultiTaskCommunityModel, self).__init__()
        
        self.bert = AutoModel.from_pretrained(model_name)
        self.config = self.bert.config
        self.dropout = nn.Dropout(0.2)
        
        # 多任务分类头
        self.category_classifier = nn.Linear(self.config.hidden_size, num_categories)
        self.emotion_classifier = nn.Linear(self.config.hidden_size, num_emotions)
        self.urgency_classifier = nn.Linear(self.config.hidden_size, num_urgency)
        
    def forward(self, input_ids, attention_mask, category_labels=None, emotion_labels=None, urgency_labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]
        
        pooled_output = self.dropout(pooled_output)
        
        category_logits = self.category_classifier(pooled_output)
        emotion_logits = self.emotion_classifier(pooled_output)
        urgency_logits = self.urgency_classifier(pooled_output)
        
        loss = None
        if category_labels is not None and emotion_labels is not None and urgency_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            category_loss = loss_fct(category_logits, category_labels)
            emotion_loss = loss_fct(emotion_logits, emotion_labels)
            urgency_loss = loss_fct(urgency_logits, urgency_labels)
            
            loss = (1.0 * category_loss + 0.8 * emotion_loss + 0.6 * urgency_loss)
        
        return {
            'loss': loss,
            'category_logits': category_logits,
            'emotion_logits': emotion_logits,
            'urgency_logits': urgency_logits
        }

def load_and_preprocess_data(csv_path):
    """加载和预处理数据 - 修复版本"""
    print("正在加载数据...")
    df = pd.read_csv(csv_path)
    print(f"数据加载完成，共 {len(df)} 条样本")
    df = df[df['text'] != 'text'] 
    df = df.dropna()
    df = df[df['text'].str.len() > 0]
    print(f"清洗后数据: {len(df)} 条")
    print("\n=== 真实数据分布 ===")
    print("类别分布:")
    print(df['category'].value_counts())
    print("\n情感分布:")
    print(df['emotion'].value_counts())
    print("\n紧急程度分布:")
    print(df['urgency'].value_counts())
    category_unique = sorted([x for x in df['category'].unique() if x != 'category'])
    emotion_unique = sorted([x for x in df['emotion'].unique() if x != 'emotion']) 
    urgency_unique = sorted([x for x in df['urgency'].unique() if x != 'urgency'])
    
    category_mapping = {label: idx for idx, label in enumerate(category_unique)}
    emotion_mapping = {label: idx for idx, label in enumerate(emotion_unique)}
    urgency_mapping = {label: idx for idx, label in enumerate(urgency_unique)}
    
    print(f"\n=== 正确的标签映射 ===")
    print(f"类别 ({len(category_mapping)}种): {category_mapping}")
    print(f"情感 ({len(emotion_mapping)}种): {emotion_mapping}")
    print(f"紧急程度 ({len(urgency_mapping)}种): {urgency_mapping}")
    label_mappings = {
        'category': category_mapping,
        'emotion': emotion_mapping,
        'urgency': urgency_mapping
    }
    
    with open('label_mappings.json', 'w', encoding='utf-8') as f:
        json.dump(label_mappings, f, ensure_ascii=False, indent=2)
    valid_data = df[
        (df['category'].isin(category_mapping.keys())) & 
        (df['emotion'].isin(emotion_mapping.keys())) & 
        (df['urgency'].isin(urgency_mapping.keys()))
    ]
    
    texts = valid_data['text'].tolist()
    category_numeric = [category_mapping[cat] for cat in valid_data['category']]
    emotion_numeric = [emotion_mapping[emo] for emo in valid_data['emotion']]
    urgency_numeric = [urgency_mapping[urg] for urg in valid_data['urgency']]
    
    print(f"\n有效训练数据: {len(texts)} 条")
    return texts, category_numeric, emotion_numeric, urgency_numeric, category_mapping, emotion_mapping, urgency_mapping

def main():
    # 配置参数
    MODEL_NAME =r"D:\work\community_ai\chinese_macbert_base"
    DATA_PATH = "dataset.csv"
    
    BATCH_SIZE = 16
    LEARNING_RATE = 2e-5
    EPOCHS = 20           # 设置较大的epoch数
    PATIENCE = 3          # 早停耐心值
    MAX_LENGTH = 128
    SAVE_DIR = "./community_model"
    
    print("=== 社区工单多任务分类模型训练 ===")
    print(f"基础模型: {MODEL_NAME}")
    
    # 1. 加载tokenizer - 添加重试机制
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        print("✓ Tokenizer加载完成")
    except Exception as e:
        print(f"❌ Tokenizer加载失败: {e}")
        print("尝试使用本地缓存...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
        print("✓ 使用本地缓存Tokenizer")
    
    # 2. 加载和预处理数据
    texts, category_labels, emotion_labels, urgency_labels, category_mapping, emotion_mapping, urgency_mapping = load_and_preprocess_data(DATA_PATH)
    
    if len(texts) == 0:
        print("❌ 没有有效数据，请检查CSV文件格式")
        return
    
    # 3. 创建数据集
    dataset = CommunityDataset(texts, category_labels, emotion_labels, urgency_labels, tokenizer, MAX_LENGTH)
    
    # 分割训练集和验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"\n数据集分割:")
    print(f"训练集: {train_size} 条")
    print(f"验证集: {val_size} 条")
    
    # 4. 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 5. 初始化模型 - 添加重试机制
    try:
        model = MultiTaskCommunityModel(
            model_name=MODEL_NAME,
            num_categories=len(category_mapping),
            num_emotions=len(emotion_mapping),
            num_urgency=len(urgency_mapping)
        ).to(device)
        print("✓ 模型加载完成")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        print("尝试使用本地缓存...")
        model = MultiTaskCommunityModel(
            model_name=MODEL_NAME,
            num_categories=len(category_mapping),
            num_emotions=len(emotion_mapping),
            num_urgency=len(urgency_mapping)
        ).to(device)
        print("✓ 使用本地缓存模型")
    
    print(f"\n模型配置:")
    print(f"类别数: {len(category_mapping)}")
    print(f"情感数: {len(emotion_mapping)}")
    print(f"紧急程度数: {len(urgency_mapping)}")
    
    # 6. 设置优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    # 7. 训练循环
    print("\n开始训练...")
    best_val_accuracy = 0
    patience_counter = 0
    training_history = []
    
    for epoch in range(EPOCHS):
        # 训练阶段
        model.train()
        total_train_loss = 0
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS} [训练]')
        
        for batch in train_bar:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            cat_labels = batch['category_labels'].to(device)
            emo_labels = batch['emotion_labels'].to(device)
            urg_labels = batch['urgency_labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                category_labels=cat_labels,
                emotion_labels=emo_labels,
                urgency_labels=urg_labels
            )
            
            loss = outputs['loss']
            total_train_loss += loss.item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            train_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        # 验证阶段
        model.eval()
        category_correct = 0
        emotion_correct = 0
        urgency_correct = 0
        total_samples = 0
        
        val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{EPOCHS} [验证]')
        
        with torch.no_grad():
            for batch in val_bar:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                cat_labels = batch['category_labels'].to(device)
                emo_labels = batch['emotion_labels'].to(device)
                urg_labels = batch['urgency_labels'].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    category_labels=cat_labels,
                    emotion_labels=emo_labels,
                    urgency_labels=urg_labels
                )
                
                category_preds = torch.argmax(outputs['category_logits'], dim=1)
                emotion_preds = torch.argmax(outputs['emotion_logits'], dim=1)
                urgency_preds = torch.argmax(outputs['urgency_logits'], dim=1)
                
                category_correct += (category_preds == cat_labels).sum().item()
                emotion_correct += (emotion_preds == emo_labels).sum().item()
                urgency_correct += (urgency_preds == urg_labels).sum().item()
                total_samples += cat_labels.size(0)
                
                current_acc = (category_correct + emotion_correct + urgency_correct) / (3 * total_samples)
                val_bar.set_postfix({'avg_acc': f'{current_acc:.3f}'})
        
        category_acc = category_correct / total_samples
        emotion_acc = emotion_correct / total_samples
        urgency_acc = urgency_correct / total_samples
        avg_acc = (category_acc + emotion_acc + urgency_acc) / 3
        
        print(f"\nEpoch {epoch+1} 结果:")
        print(f"  训练损失: {avg_train_loss:.4f}")
        print(f"  类别准确率: {category_acc:.4f}")
        print(f"  情感准确率: {emotion_acc:.4f}")
        print(f"  紧急程度准确率: {urgency_acc:.4f}")
        print(f"  平均准确率: {avg_acc:.4f}")
        
        # 早停机制
        if avg_acc > best_val_accuracy:
            best_val_accuracy = avg_acc
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': best_val_accuracy,
                'category_mapping': category_mapping,
                'emotion_mapping': emotion_mapping,
                'urgency_mapping': urgency_mapping
            }, 'best_community_model.pth')
            print(f"  ✅ 保存最佳模型 (准确率: {best_val_accuracy:.4f})")
        else:
            patience_counter += 1
            print(f"  ⏳ 早停计数: {patience_counter}/{PATIENCE}")
            
            if patience_counter >= PATIENCE:
                print(f"  🛑 早停触发! 在 Epoch {epoch+1} 停止训练")
                break
    
    print(f"\n=== 训练完成 ===")
    print(f"最终最佳验证准确率: {best_val_accuracy:.4f}")
    print(f"总共训练轮数: {epoch + 1}")
    
    # 保存最终模型
    model.bert.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)
    print(f"模型已保存到: {SAVE_DIR}")

if __name__ == "__main__":
    main()
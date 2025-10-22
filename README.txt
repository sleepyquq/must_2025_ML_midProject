# MNIST Pairwise Comparison Project

## 📋 项目概述
这是一个基于MNIST数据集的数字比较机器学习项目，旨在判断左右两个数字的大小关系（左>右或左<右）。

## 🎯 项目目标
- 实现端到端的数字比较模型
- 达到超过70%的准确率目标
- 处理包含遮挡和扰动的测试数据

## 📊 性能成果
- **最佳验证准确率**: 85.17% (远超70%目标)
- **性能提升幅度**: +15.17%
- **模型稳定性**: 训练过程收敛稳定

## 📁 项目文件结构
must_2025_ML_midProject/
├── material.txt # 材料下载
├── 📂 src/ # 源代码文件夹
├── 📄 best_occlusion_model.pth # 最佳遮挡优化模型权重 (10MB)
├── 📄 final_model_weights.pth # 最终模型权重文件 (10MB)
├── 📄 training_history.json # 训练历史记录 (4KB)
├── 📄 error_analysis.json # 错误分析报告 (1KB)
├── 📄 pred_private.csv # 私有测试集预测结果 (110KB)
├── 📄 pred_public.csv # 公开测试集预测结果 (28KB)
└── 📄 project_summary.md # 项目总结文档 (1KB)
## 🔧 技术实现

### 模型架构
- **网络类型**: 自定义CNN with遮挡鲁棒性优化
- **卷积层**: 多层卷积+池化结构
- **正则化**: Dropout + BatchNorm
- **激活函数**: ReLU

### 训练策略
- **优化器**: AdamW with权重衰减
- **学习率调度**: ReduceLROnPlateau
- **数据增强**: 旋转、平移、遮挡模拟
- **训练轮次**: 80 epochs

## 🚀 快速开始

### 环境要求
bash
pip install torch torchvision numpy pandas matplotlib scikit-learn
### 加载模型进行预测
python
import torch
import torch.nn as nn
定义模型架构（需要与训练时一致）
class OcclusionRobustCNN(nn.Module):
def init(self):
super(OcclusionRobustCNN, self).init()
# 您的模型定义代码...
def forward(self, x):
    # 前向传播代码...
    return x
加载训练好的模型
model = OcclusionRobustCNN()
model.load_state_dict(torch.load('best_occlusion_model.pth'))
model.eval()
进行预测
def predict_comparison(left_digit, right_digit):
# 预处理和预测逻辑...
prediction = model(processed_input)
return 'Left > Right' if prediction > 0.5 else 'Left < Right'
## 📈 性能分析

### 训练过程
- 详细训练历史见: `training_history.json`
- 错误分析报告见: `error_analysis.json`

### 关键指标
| 指标 | 数值 | 状态 |
|------|------|------|
| 目标准确率 | 70.00% | ✅ 基准 |
| 达成准确率 | 85.17% | ✅ 超额完成 |
| 性能提升 | +15.17% | 🎉 优秀 |

## 📤 预测结果

### 测试集预测
- **公开测试集**: `pred_public.csv` (28KB)
- **私有测试集**: `pred_private.csv` (110KB) - **最终提交文件**

### 文件格式
csv
id,label
0,1
1,0
2,1
...
## 🔍 详细分析

### 错误分析
项目包含完整的错误分析，重点关注：
- 遮挡导致的分类错误
- 模型置信度分布
- 各类别平衡性

### 模型比较
- `best_occlusion_model.pth`: 针对遮挡优化的最佳模型
- `final_model_weights.pth`: 最终训练完成的模型权重

## 📚 相关文档

- **项目总结**: `project_summary.md` - 完整项目报告
- **训练历史**: `training_history.json` - 详细训练过程记录
- **错误分析**: `error_analysis.json` - 模型错误模式分析

## 👥 使用说明

### 研究人员
1. 查看 `project_summary.md` 了解项目全貌
2. 分析 `training_history.json` 研究训练过程
3. 参考 `error_analysis.json` 了解模型局限性

### 开发者
1. 使用 `best_occlusion_model.pth` 进行推理
2. 参考预测文件格式生成新预测
3. 基于现有代码进行模型改进

## 🎯 项目亮点

1. **高性能**: 85.17%准确率远超目标要求
2. **鲁棒性**: 专门针对遮挡优化，适应真实场景
3. **完整性**: 包含训练、评估、预测全流程
4. **可复现**: 详细文档确保结果可复现


---

**最后更新**: 2025/10/22  
**项目状态**: ✅ 已完成并验证通过
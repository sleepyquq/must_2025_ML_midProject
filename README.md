# MNIST Pairwise Comparison Project 

## 📋 Project Overview
This project implements a deep learning-based MNIST digit comparison system that can determine the size relationship between two digits (left > right or left < right). The project adopts an end-to-end training approach, learning the comparison relationship directly from pixel data, and is specifically optimized for disturbances like occlusion in the test set.

## 🎯 Project Goals
- Implement an end-to-end digit comparison model with accuracy exceeding 70%
- Handle test data containing rotation, translation, noise, and occlusion
- Provide complete reproducible code and environment configuration

## 📊 Performance Achievements
- **Best Validation Accuracy**: 85.17% (significantly exceeding the 70% target)
- **Performance Improvement**: +15.17%
- **Model Stability**: Stable training process convergence with a standard deviation of only 0.0028

## 📁 Project File Structure
must_2025_ML_midProject/  
├── 📂 src/ # Source code directory  
│    ├── 📄 mlMid2.ipynb # Complete project code    
│    ├── 📄 mlMid2_colab.ipynb # Project run preview   
│    ├── 📄 mlMid2_en.ipynb # English version   
├── 📄 best_occlusion_model.pth # Backup of the best model weights during training (10MB)  
├── 📄 final_model_weights.pth # Final model weights file after training (10MB)  
├── 📄 training_history.json # Training history record (4KB)  
├── 📄 error_analysis.json # Error analysis report (1KB)  
├── 📄 pred_private.csv # Private test set prediction results (110KB) - Final submission file  
├── 📄 pred_public.csv # Public test set prediction results (28KB)  
├── 📄 project_summary.md # Project summary document (1KB)  
├── 📄 requirements.txt # Environment dependencies  
├── 📄 material.txt # Download links for initial files provided by the project  
└── 📄 README.md # Project description document (this file)  

## 🛠️ Environment Setup
Due to the author's computer AMD GPU limitations, the development and testing of this project were conducted entirely on colab.

### System Requirements
- Python 3.8+
- PyTorch 1.12.0+
- GPU support (recommended, but CPU can also run)

### Install Dependencies
bash
Use pip to install required packages
pip install -r requirements.txt
### requirements.txt Content
torch==1.12.1
torchvision==0.13.1
numpy==1.21.6
pandas==1.3.5
matplotlib==3.5.3
scikit-learn==1.0.2
seaborn==0.11.2
tqdm==4.64.0
Refer specifically to the code in cell 2 of `mlMid2.ipynb`. This code fully configures the environment. When testing this project on colab, simply run this code segment.

## 🚀 Quick Start
After configuring the environment, we can begin.

### 1. Data Preparation
Download the compressed package provided in `material.txt`. After extraction, you will get the data and scripts folders. The initial files required for the project are in these two folders.
Upload the initial files to colab. You can also configure Google Drive like the author did. If not needed, please ignore the code in cell 1.
When reproducing this project, pay attention to modifying the corresponding file paths in the source code, otherwise errors may occur.

### 2. Model Training
Note: Before training the model, data should be loaded. See cells 3-4.
Perform initial data preprocessing, build the model, and start training.
After the preceding code segments are executed sequentially, continue to execute the code in cells 5-7 in order. After executing the code in cell 7,
the main model training process begins, which may take some time depending on the hardware.
The models trained by the author in the early stages of the project were not ideal and suffered from overfitting. After loading the data and observing a subset of samples, it was determined that the main obstacle was data occlusion. To address this characteristic, data preprocessing, the model, and the training function were optimized. For details, refer to the comments in the corresponding parts of the source code.
The code in cells 5-7 that you see now is the optimized version.

### 3. Generate Predictions
Execute the code in cells 11-12. This will generate predictions for the public test set and the private test set respectively. The code segments have integrated the functionality of `scripts/check_submission.py`, ensuring the correct format of the prediction files.

### 4. Other
In summary, after configuring the corresponding file addresses, execute the entire source code sequentially to fully reproduce the entire project.

## ⚙️ Training Configuration

### Hyperparameter Settings
| Parameter | Value | Description |
|-----------|-------|-------------|
| Learning Rate | 0.001 | AdamW optimizer initial learning rate |
| Batch Size | 64 | Number of training samples per batch |
| Epochs | 80 | Number of complete training cycles |
| Dropout Rate | 0.5 | Regularization parameter to prevent overfitting |
| Weight Decay | 1e-4 | L2 regularization coefficient |

### Data Augmentation Strategy
- Random rotation (±10 degrees)
- Random translation (±10%)
- Random occlusion simulation (20% probability)
- Contrast and brightness adjustment

## 📈 Model Performance

### Accuracy Results
| Dataset | Sample Count | Accuracy | Note |
|---------|--------------|----------|------|
| Training Set | 50,000 | 82.97% | - |
| Validation Set | 10,000 | 85.17% | **Best Performance** |
| Public Test Set | 5,000 | 84.51% | Local Validation |

### Training Statistics
- **Model Size**: 10 MB (final weight file)
- **Training Time**: ~45 minutes (on Colab T4 GPU)
- **Parameter Count**: Approximately 1.3 million trainable parameters
- **Inference Speed**: Average 2.3ms/sample (GPU inference)

## 🔍 Error Analysis

The project includes detailed error analysis, focusing on:
- **Classification errors due to occlusion**: Error rate significantly increases for highly occluded samples.
- **Confidence distribution**: Average confidence of misclassified samples is 0.72.
- **Error types**: Analysis of false positive vs. false negative ratios.

Detailed error analysis can be found in the `error_analysis.json` file.

## 📊 Result Visualization

### Learning Curves
The training process shows a stable convergence trend:
- Training loss decreased from 0.6988 to 0.4436.
- Validation accuracy increased from 50.18% to 85.17%.
- No overfitting observed, good generalization ability.

### Confusion Matrix
The validation set confusion matrix shows balanced classification performance:
- Class 0 (left < right) accuracy: 84.92%
- Class 1 (left > right) accuracy: 85.42%

## 🎯 Technical Highlights

### Model Architecture Innovations
- **Multi-layer CNN Design**: 3 convolutional blocks + deep classifier.
- **Occlusion Robustness**: Specifically optimized for test set disturbances.
- **Adaptive Pooling**: Adapts to different input sizes.

### Training Strategy Optimization
- **Label Smoothing**: Reduces overfitting risk.
- **Gradient Clipping**: Ensures training stability.
- **Learning Rate Scheduling**: Adaptive learning rate adjustment.

### Regularization Techniques
- Batch Normalization
- Dropout (convolutional and fully connected layers)
- L2 Weight Decay

## 📋 Usage Instructions

### For Researchers
1. View `project_summary.md` for a complete project overview.
2. Analyze `training_history.json` to study the training process.
3. Refer to `error_analysis.json` to understand model limitations.

### For Developers
1. Use `best_occlusion_model.pth` for inference.
2. Refer to the prediction file format to generate new predictions.
3. Improve the model based on the existing code.

### For Evaluators
1. Verify the format correctness of `pred_private.csv`.
2. Use the provided validation script to check the submission file.
3. Refer to performance metrics for result evaluation.

## 🔧 Troubleshooting

### Common Issues
1. **Insufficient Memory**: Reduce batch size or use CPU mode.
2. **Dependency Conflicts**: Use a virtual environment to isolate package versions.
3. **Incorrect Data Path**: Ensure data files are placed in the correct directory.

### Getting Help
If encountering problems, please check:
- If the environment dependencies are completely installed.
- If the data file paths are correct.
- If file permissions are set appropriately.

## 📄 Related Documents

- **Project Summary**: `project_summary.md` - Complete project report.
- **Training History**: `training_history.json` - Detailed training process record.
- **Error Analysis**: `error_analysis.json` - Analysis of model error patterns.

## 👥 Contribution Statement

### AI Tool Usage
This project used AI tools for assistance in the following aspects:
- Code debugging and optimization suggestions.
- Documentation template generation.
- Error analysis ideas.

## 📞 Contact Information

**Project Repository**: https://github.com/sleepyquq/must_fie_2509_ML_midProject  
**Last Updated**: October 22, 2025  
**Project Status**: ✅ Completed and Verified

---

## 📌 Important Notes

1. **Final Submission File**: `pred_private.csv` (must pass format validation).
2. **Model Weights**: If training is interrupted unexpectedly, use `best_occlusion_model.pth` to obtain the best performance without retraining from scratch.
3. **Environment Consistency**: Ensure the use of the specified versions of dependency packages.
4. **Reproducibility**: All random seeds are fixed to ensure consistent results.

**Enjoy! If you have any questions, please submit an Issue or contact the project maintainer.**

---
**This project provides an English version on the basis of the Chinese original. In case of any discrepancy between the English version and the Chinese original version, the Chinese original version shall prevail.  
本项目在中文原版的基础上，提供了英文版本，如果英文版本与中文原版有出入，以中文原版为准。*  
---

# 中文版
# MNIST Pairwise Comparison Project

## 📋 项目概述
本项目实现了一个基于深度学习的MNIST数字比较系统，能够判断左右两个数字的大小关系（左>右或左<右）。项目采用端到端的训练方式，直接从像素数据学习比较关系，并针对测试集中的遮挡等扰动进行了专门优化。

## 🎯 项目目标
- 实现端到端的数字比较模型，准确率超过70%
- 处理包含旋转、平移、噪声和遮挡的测试数据
- 提供完整的可复现代码和环境配置

## 📊 性能成果
- **最佳验证准确率**: 85.17% (远超70%目标要求)
- **性能提升幅度**: +15.17%
- **模型稳定性**: 训练过程收敛稳定，标准差仅0.0028

## 📁 项目文件结构
must_2025_ML_midProject/  
├── 📂 src/ # 源代码目录  
│    ├── 📄 mlMid2.ipynb # 项目完整代码    
│    ├── 📄 mlMid2_colab.ipynb # 项目运行预览    
│    ├── 📄 mlMid2_en.ipynb # English version    
├── 📄 best_occlusion_model.pth # 训练过程中最佳模型权重的备份(10MB)  
├── 📄 final_model_weights.pth # 训练完成后最终模型权重文件 (10MB)  
├── 📄 training_history.json # 训练历史记录 (4KB)  
├── 📄 error_analysis.json # 错误分析报告 (1KB)  
├── 📄 pred_private.csv # 私有测试集预测结果 (110KB) - 最终提交文件  
├── 📄 pred_public.csv # 公开测试集预测结果 (28KB)  
├── 📄 project_summary.md # 项目总结文档 (1KB)  
├── 📄 requirements.txt # 环境依赖  
├── 📄 material.txt # 项目提供的初始文件下载链接  
└── 📄 README.md # 项目说明文档 (本文件)  

## 🛠️ 环境设置
由于作者电脑AMD GPU的限制，故本项目的开发与测试均在colab上进行

### 系统要求
- Python 3.8+
- PyTorch 1.12.0+
- GPU支持 (推荐，但CPU也可运行)

### 安装依赖  
bash  
使用pip安装所需包  
pip install -r requirements.txt  
### requirements.txt 内容  
torch==1.12.1  
torchvision==0.13.1  
numpy==1.21.6  
pandas==1.3.5  
matplotlib==3.5.3  
scikit-learn==1.0.2  
seaborn==0.11.2  
tqdm==4.64.0  
具体参见`mlMid2.ipynb`中单元格2的代码，这段代码充分配置了环境，在colab上测试本项目时，直接运行该段代码  

## 🚀 快速开始  
配置环境后，我们可以开始了  

### 1. 数据准备
下载`material.txt`中提供的压缩包，解压后得到data以及scripts文件夹，项目需要的初始文件都在这两个文件夹中  
上传初始文件至colab，您也可以像作者一样配置Google drive，如不需要，请忽略单元格1代码  
复现本项目时，注意修改源代码中对应文件的地址，否则可能报错

### 2. 模型训练
注意，训练模型前应加载数据，参见单元格3-4  
对数据进行初步预处理，构建模型，开始训练。  
上述步骤在前序代码依次执行完毕后，继续按顺序依次执行单元格5-7代码，其中执行单元格7代码后  
进入主要的模型训练过程，可能需要花费一定的时间，具体取决于硬件  
作者在项目前期训练出来的模型并不理想，发生了过拟合现象，通过加载数据并抽取一部分样本观察后，判断主要障碍是数据被遮挡，针对这个特点，优化了数据的预处理和模型以及训练函数，具体参见源代码对应部分的注释  
现在看到的单元格5-7代码是优化后的版本。

### 3. 生成预测
执行单元格11-12代码，将分别生成公开测试集与私有测试集的预测，并且代码段中已经集成了`scripts/check_submission.py`的功能，保证了预测文件格式的正确  

### 4. 其他  
总而言之，在配置好相应文件地址后，按顺序依次执行整段源代码，即可完整复现整个项目
 
## ⚙️ 训练配置

### 超参数设置
| 参数 | 值 | 说明 |
|------|-----|------|
| 学习率 | 0.001 | AdamW优化器初始学习率 |
| 批大小 | 64 | 每批次训练样本数 |
| 训练轮次 | 80 | 完整训练周期数 |
| Dropout率 | 0.5 | 防止过拟合的正则化参数 |
| 权重衰减 | 1e-4 | L2正则化系数 |

### 数据增强策略
- 随机旋转 (±10度)
- 随机平移 (±10%)
- 随机遮挡模拟 (20%概率)
- 对比度和亮度调整

## 📈 模型性能

### 准确率结果
| 数据集 | 样本数量 | 准确率 | 备注 |
|--------|----------|--------|------|
| 训练集 | 50,000 | 82.97% | - |
| 验证集 | 10,000 | 85.17% | **最佳性能** |
| 公开测试集 | 5,000 | 84.51% | 本地验证 |

### 训练统计
- **模型大小**: 10 MB (最终权重文件)
- **训练时间**: ~45分钟 (在Colab T4 GPU上)
- **参数量**: 约1.3百万可训练参数
- **推理速度**: 平均2.3ms/样本 (GPU推理)

## 🔍 错误分析

项目包含详细的错误分析，重点关注：
- **遮挡导致的分类错误**: 高遮挡样本错误率显著提升
- **置信度分布**: 错误分类样本平均置信度0.72
- **错误类型**: 假阳性 vs 假阴性比例分析

详细错误分析见 `error_analysis.json` 文件。

## 📊 结果可视化

### 学习曲线
训练过程显示稳定的收敛趋势：
- 训练损失从0.6988下降至0.4436
- 验证准确率从50.18%提升至85.17%
- 无过拟合现象，泛化能力良好

### 混淆矩阵
验证集混淆矩阵显示均衡的分类性能：
- 类别0 (左<右) 准确率: 84.92%
- 类别1 (左>右) 准确率: 85.42%

## 🎯 技术亮点

### 模型架构创新
- **多层CNN设计**: 3个卷积块 + 深层分类器
- **遮挡鲁棒性**: 专门针对测试集扰动优化
- **自适应池化**: 适应不同输入尺寸

### 训练策略优化
- **标签平滑**: 减少过拟合风险
- **梯度裁剪**: 确保训练稳定性  
- **学习率调度**: 自适应调整学习步长

### 正则化技术
- Batch Normalization
- Dropout (卷积层和全连接层)
- L2权重衰减

## 📋 使用说明

### 研究人员
1. 查看 `project_summary.md` 了解项目全貌
2. 分析 `training_history.json` 研究训练过程
3. 参考 `error_analysis.json` 了解模型局限性

### 开发者
1. 使用 `best_occlusion_model.pth` 进行推理
2. 参考预测文件格式生成新预测
3. 基于现有代码进行模型改进

### 评估人员
1. 验证 `pred_private.csv` 格式正确性
2. 使用提供的验证脚本检查提交文件
3. 参考性能指标进行结果评估

## 🔧 故障排除

### 常见问题
1. **内存不足**: 减小批大小或使用CPU模式
2. **依赖冲突**: 使用虚拟环境隔离包版本
3. **数据路径错误**: 确保数据文件放置在正确目录

### 获取帮助
如遇问题，请检查：
- 环境依赖是否完整安装
- 数据文件路径是否正确
- 文件权限是否设置适当

## 📄 相关文档

- **项目总结**: `project_summary.md` - 完整项目报告
- **训练历史**: `training_history.json` - 详细训练过程记录  
- **错误分析**: `error_analysis.json` - 模型错误模式分析


## 👥 贡献说明

### AI工具使用
本项目在以下方面使用了AI工具辅助：
- 代码调试和优化建议
- 说明文档模板生成
- 错误分析思路

## 📞 联系信息

**项目仓库**: https://github.com/sleepyquq/must_fie_2509_ML_midProject  
**最后更新**: 2025年10月22日  
**项目状态**: ✅ 已完成并验证通过

---

## 📌 重要提示

1. **最终提交文件**: `pred_private.csv` (必须通过格式验证)
2. **模型权重**: 如果训练意外中断，使用 `best_occlusion_model.pth` 获得最佳性能，而不必从头开始训练
3. **环境一致性**: 确保使用指定版本的依赖包
4. **可复现性**: 所有随机种子已固定，确保结果一致

**祝您使用愉快！如有问题欢迎提交Issue或联系项目维护者。**








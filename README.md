# LGH-SAM

LGH-SAM 是基于 Segment Anything Model（SAM）的医学图像分割模型，通过局部与全局特征交互、轻量特征适配及先验引导的两阶段解码生成分割预测。

本目录包含模型定义、训练与评估逻辑，以及数据处理、损失和指标计算工具。

## 训练权重下载（Trained checkpoints）

- **百度网盘（Baidu Netdisk）**：[LGH-SAM](https://pan.baidu.com/s/1UXz7f4vJ_4xN42yHctH3Dg)
- **提取码（Access code）**：`evig`

打开链接并输入提取码，下载所需任务的训练权重。如果下载的是压缩包，请先解压。评估时将 `dir_checkpoint` 指向包含 `checkpoint_best.pth` 的目录。同一目录如有 `args.json`，请一并保留，`val.py` 会用它恢复部分训练配置。

评估还需要对应任务的 `cfg.py` 配置、SAM 基座权重（`sam_ckpt`）和数据路径。模型架构与类别数须与训练时一致。训练完成的 LGH-SAM 权重和用于构建模型的 SAM 基座权重是两个不同的文件。

## 模型结构

模型的主要数据流为：

```text
输入图像
   │
   ├── ViT 编码分支 ── 全局上下文特征 ──┐
   │                                  ├── 特征交互与融合
   └── CNN 局部分支 ── 局部空间特征 ────┘          │
                                                ▼
                                      第一阶段掩膜解码器
                                                │
                          初步预测、掩膜特征、像素解码特征
                                                │
                                                ▼
                                  先验引导的第二阶段解码器
                                                │
                                             精细预测
                                                │
                           与第一阶段预测的类别概率进行融合
                                                │
                                                ▼
                                           最终分割结果
```

SAM 的提示编码器提供位置编码和提示嵌入接口；当前 `train.py`、`val.py` 中的分割流程使用无外部点、框和掩膜提示的嵌入。

### 局部与全局特征编码

`ImageEncoderViT` 提取图像的全局上下文表示，`SingleCNNEmbed` 构建并行 CNN 分支，提取局部空间信息。全局注意力编码块中的 `qkvAttention` / `CBA_Adapter` 将两路特征进行交互，CNN 特征还参与编码器输出前的融合。

`Adapter` 采用降维、非线性变换和升维的瓶颈结构，并按所在模块配置残差连接。编码器和第一阶段解码器中的适配层分别定义在 `image_encoder.py` 与 `transformer.py` 中。

### 先验引导的两阶段解码

第一阶段 `MaskDecoder` 生成初步掩膜，并输出供第二阶段使用的中间特征。

第二阶段 `MaskDecoder2` 接收图像嵌入、第一阶段预测及中间特征：`CMAttn` 用于图像嵌入增强，`Attention2` 在交叉注意力中引入掩膜先验，分层像素解码部分通过上采样和跨阶段跳跃连接生成精细预测。

完整分割流程将两阶段输出对齐后，分别转换为类别概率，再以等权方式融合。**两阶段流程由 `train.py` 和 `val.py` 显式串联；`sam.py` 提供模型组件容器及保留的 SAM 接口。**

## 目录结构

```text
.
├── README.md                         # 项目结构说明
├── train.py                          # 训练流程、两阶段前向与监督逻辑
├── val.py                            # 权重加载、两阶段预测与指标汇总
├── models/
│   ├── sam/                          # LGH-SAM 的主要模型实现
│   │   ├── __init__.py                # 模型构建接口与注册表导出
│   │   ├── build_sam.py               # 编码器、提示编码器与解码器的组装
│   │   ├── modeling/
│   │   │   ├── sam.py                # SAM 模型容器与基础接口
│   │   │   ├── image_encoder.py      # ViT、CNN 分支、Adapter 与特征交互
│   │   │   ├── prompt_encoder.py     # 提示与位置编码
│   │   │   ├── mask_decoder.py       # 两阶段掩膜解码与先验注意力
│   │   │   ├── transformer.py        # 第一阶段双向 Transformer 与 Adapter
│   │   │   ├── common.py             # Adapter、MLP 与归一化基础模块
│   │   │   ├── tiny_vit_sam.py       # TinyViT 编码器定义
│   │   │   └── vit.py                # 附带的 ViT 组件
│   │   ├── predictor.py              # 保留的 SAM 预测接口
│   │   ├── automatic_mask_generator.py
│   │   │                             # 保留的 SAM 自动掩膜生成接口
│   │   └── utils/                    # 几何变换、掩膜处理与 ONNX 辅助功能
│   ├── sam_LoRa.py                    # LoRA 适配相关模型定义
│   ├── unet/                         # 附带的 U-Net 网络定义
│   ├── tag/                          # 附带的 Transformer 网络组件
│   └── ...                           # ResNet、VGG、EfficientNet 等保留网络
└── utils/
    ├── dataset.py                    # 图像、掩膜及数据清单读取与变换
    ├── losses.py                     # 损失函数
    ├── dsc.py                        # Dice 指标计算
    ├── medical_metrics.py            # 像素准确率与 HD95 等指标
    ├── funcs.py                      # 掩膜、区域及提示相关辅助函数
    └── utils.py                      # 可视化及通用辅助功能
```

## 核心模块与源码对应

| 模块 | 主要类或对象 | 文件 |
|---|---|---|
| 模型构建 | `sam_model_registry`、`_build_sam` | [build_sam.py](models/sam/build_sam.py) |
| 模型容器 | `Sam` | [sam.py](models/sam/modeling/sam.py) |
| 图像编码与局部特征分支 | `ImageEncoderViT`、`SingleCNNEmbed`、`Block` | [image_encoder.py](models/sam/modeling/image_encoder.py) |
| 跨分支特征交互 | `qkvAttention`、`CBA_Adapter` | [image_encoder.py](models/sam/modeling/image_encoder.py) |
| 轻量特征适配 | `Adapter` | [common.py](models/sam/modeling/common.py) |
| 提示编码 | `PromptEncoder` | [prompt_encoder.py](models/sam/modeling/prompt_encoder.py) |
| 第一阶段掩膜预测 | `MaskDecoder`、`TwoWayTransformer` | [mask_decoder.py](models/sam/modeling/mask_decoder.py)、[transformer.py](models/sam/modeling/transformer.py) |
| 先验引导与掩膜注意力 | `CMAttn`、`Attention2`、`TwoWayTransformer2` | [mask_decoder.py](models/sam/modeling/mask_decoder.py) |
| 第二阶段与分层像素解码 | `MaskDecoder2`、`output_upscaling`、`skip_connect` | [mask_decoder.py](models/sam/modeling/mask_decoder.py) |
| 两阶段前向及概率融合 | `hsam_forward`、`hsam_average_prob` 与评估前向逻辑 | [train.py](train.py)、[val.py](val.py) |

`models/` 下的其他网络属于保留的模型或辅助组件，不代表 LGH-SAM 默认使用这些网络作为主干。`predictor.py` 和 `automatic_mask_generator.py` 是保留的 SAM 接口，与训练、评估文件中显式定义的两阶段流程分别说明。

ViT-B、ViT-L、ViT-H 的构建路径在 `build_sam.py` 的 `_build_sam` 中组装两阶段模型；TinyViT 构建入口保留其原有接口，不将其直接等同于上述完整结构。

## 代码范围与来源

本目录提供 LGH-SAM 的核心模型源码及相关训练、评估实现。训练权重通过上面的网盘链接提供。当前目录缺少数据集和训练与评估脚本使用的 `cfg.py`。

模型建立在 SAM 等已有组件之上，源码保留相应的原始版权和来源信息。目录中的基础模型、适配模块和其他附带网络应按各自实际来源区分。

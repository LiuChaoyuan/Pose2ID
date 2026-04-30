# IPG 改进说明

本文档说明本次在 Pose2ID 的 IPG 部分加入的三个可消融模块，以及训练、推理时的启动方式。

## 1. 模块开关

训练配置位于 `IPG/configs/train_ipg.yaml`，推理配置位于 `IPG/configs/inference.yaml`。新增配置统一放在 `features` 下：

```yaml
features:
  part_reference_bank:
    enabled: true
    lambda_init: 0.0
    masks:
      ref_upper_root: null
      ref_lower_root: null
      target_upper_root: null
      target_lower_root: null
  color_structure:
    enabled: true
    color_json_path: "/root/autodl-fs/datasets/market1501/clothing_colors_nl.json"
    clip_model_path: null
    color_scale: 0.25
  pose_loss:
    enabled: true
    weight: 0.05
    max_sigma: 0.35
```

`enabled` 可单独控制每个模块，便于消融实验。`clip_model_path: null` 时默认复用 `base_model_path` 下 Stable Diffusion 的 tokenizer 和 text encoder。

## 2. Part-Aware Reference Bank

原始 IPG 只把完整参考图写入一个 reference bank。本次将 bank 扩展为三类：

- `global`：完整参考图 latent。
- `upper`：参考图乘以上半身 mask 后的 latent。
- `lower`：参考图乘以下半身 mask 后的 latent。

训练和推理时，第一次 denoising timestep 会依次写入三类 bank。denoising UNet 读取时按目标区域 mask 融合：

```text
F_ref = F_global
    + lambda_upper * M_upper * (F_upper - F_global)
    + lambda_lower * M_lower * (F_lower - F_global)
```

`lambda_upper/lambda_lower` 是可学习参数，保存在 `part_bank_fusion.pth`。初始值为 `0`，因此初始行为退化为原始 IPG 的单 bank 逻辑。

如果没有 SAM3/human parsing 生成的 mask，代码会自动使用 soft 上下半身 fallback。target mask 优先使用配置路径中的 mask，缺失时根据 pose 图非零区域生成粗 mask。

## 3. 颜色结构信息注入

颜色描述来自 JSON：

```json
{
  "0002_c1s1_000451_03.jpg": {
    "upper": "A person dressed in red upper-body clothing",
    "lower": "A person wearing blue lower clothing"
  }
}
```

训练 dataset 会按参考图 basename 读取 `upper/lower` 文本。文本经冻结 CLIP text encoder 编码后，在 denoising UNet 的 cross-attention 中作为额外颜色条件注入：

- upper 文本输出由 `target_upper_mask` 控制空间作用区域。
- lower 文本输出由 `target_lower_mask` 控制空间作用区域。
- `color_scale` 控制颜色注入强度，默认 `0.25`。

缺少 JSON 或缺少某张图的记录时，颜色 token 自动置零，不影响旧数据集运行。

## 4. 可微近似姿势损失

训练中保留原始扩散 MSE 作为 `loss_mse`，并在低噪声 timestep 上额外计算姿势 proxy loss：

1. 根据 `model_pred` 和 scheduler 从 noisy latent 估计 `x0_pred`。
2. 通过冻结 VAE decode 得到可反传的近似生成图。
3. 对生成图提取 soft edge distance field。
4. 对 target pose 图生成 soft distance field。
5. 计算二者 MSE，并按 `pose_loss.weight` 加到总 loss。

只有当当前样本的 `sigma <= pose_loss.max_sigma` 时才计算，降低训练开销。

具体实现不是用 `t / num_timesteps` 这种“步数比例”，而是用 scheduler 的真实噪声强度：

```python
sigma = sqrt(1 - alphas_cumprod[t])
selected = sigma <= max_sigma
```

也就是说：只有当当前训练 batch 里某些样本的噪声强度 `sigma` 小于配置阈值 `features.pose_loss.max_sigma` 时，才会对这些低噪声样本计算姿势损失；如果一个 batch 没有满足条件的样本，就返回 0，不额外 decode。位置在 [pose_loss.py](E:/codes/Python/Pose2ID/IPG/src/models/pose_loss.py:117)。

训练总损失是在 [train_ipg.py](E:/codes/Python/Pose2ID/IPG/train_ipg.py:790) 里合成的：

```python
loss = loss_mse + pose_loss_fn.weight * loss_pose
```

姿势损失流程是：从 `model_pred + noisy_latents + timestep` 估计 `x0_latents`，用 VAE decode 得到近似生成图，再计算 soft edge / soft distance field 与 target pose distance field 的 MSE。这个实现对应你说的“扩散模型训练过程中采样噪声比例小于阈值 -> 计算姿势损失”，只是阈值判断用的是更贴近扩散噪声本身的 `sigma`。

## 5. 启动方式

离线提取 SAM3 语义掩码：

```bash
cd /root/Pose2ID/IPG
python tools/extract_sam3_part_masks.py \
  --data_root /root/autodl-fs/datasets/market1501 \
  --splits bounding_box_train bounding_box_test query \
  --standard_pose_dir ./standard_poses \
  --output_root /root/autodl-fs/datasets/market1501/sam3_part_masks
```

输出目录结构：

```text
/root/autodl-fs/datasets/market1501/sam3_part_masks/
  bounding_box_train/upper/*.jpg
  bounding_box_train/lower/*.jpg
  bounding_box_test/upper/*.jpg
  bounding_box_test/lower/*.jpg
  query/upper/*.jpg
  query/lower/*.jpg
  standard_poses/upper/*.jpg
  standard_poses/lower/*.jpg
```

如果当前环境没有 SAM3 支持，需要先在离线提取环境中安装 `IPG/tools/requirements_sam3.txt`，或安装提供 `Sam3Model` / `Sam3Processor` 的新版 `transformers`。

训练：

```bash
cd /root/Pose2ID/IPG
accelerate launch train_ipg.py --config ./configs/train_ipg.yaml
```

推理生成 Market1501：

```bash
cd /root/Pose2ID/IPG
python Market_gen.py \
  --config ./configs/inference.yaml \
  --ckpt_dir /root/autodl-fs/epoch-10000/checkpoint-10000 \
  --color_json /root/autodl-fs/datasets/market1501/clothing_colors_nl.json \
  --mask_root /root/autodl-fs/datasets/market1501/sam3_part_masks
```

常用消融：

```bash
python Market_gen.py --disable_part_bank
python Market_gen.py --disable_color_structure
```

训练阶段消融直接修改 YAML 中对应 `enabled` 字段即可。

## 6. 新增/修改文件

- `IPG/src/models/mutual_self_attention.py`：三路 reference bank、mask 融合、颜色 cross-attention 注入。
- `IPG/src/models/color_condition.py`：冻结 CLIP 文本编码器。
- `IPG/src/models/pose_loss.py`：可微近似姿势损失。
- `IPG/src/utils/mask_utils.py`：mask fallback、pose 粗 mask、attention token 对齐。
- `IPG/tools/extract_sam3_part_masks.py`：SAM3 离线提取上/下半身语义 mask。
- `IPG/src/ipg_dataset.py`：返回 mask 与颜色文本。
- `IPG/train_ipg.py`：训练链路、损失、checkpoint 保存。
- `IPG/src/pipelines/pipeline.py` 和 `IPG/Market_gen.py`：推理链路。

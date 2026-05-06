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
    # ↓↓↓ 控制颜色 prompt 的"侵入度"，详见 §3.1
    timestep_max_sigma: 0.5      # 推理：仅在 sigma <= 0.5 的低噪声步注入颜色
    min_layer_weight: 0.3        # 仅在 attn_weight >= 0.3 的中高分辨率层注入
    hard_query_gating: true      # 颜色 attn 只对 mask 内 query 生效（路径 4）
    query_threshold: 0.5         # 上面 hard mask 的 0/1 阈值
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

## 3.1 控制颜色 prompt 对图像的影响（路径 2/3/4）

颜色文本经过 SD 1.5 的 `attn2`，本质上复用了"文本→形状"的预训练能力，加之 CFG 还会把"加了颜色 prompt 的扰动"再放大 `guidance_scale` 倍，导致颜色描述很容易喧宾夺主、扰乱姿势与肢体几何。本次新增三段并联门控，**默认推理全开、训练全关**（让 attn2 学到完整颜色概念，再在推理时收敛），三段都可以通过 yaml 单独消融。

### 3.1.1 Timestep gating（路径 2，训练 + 推理均可生效）

仅在低噪声步注入颜色，让高噪声步专注于结构生成。

```yaml
features.color_structure.timestep_max_sigma: 0.5   # 推理推荐 0.5；训练默认 -1（关）
```

- 推理 loop 每步开始时计算 `sigma = sqrt(1 - alphas_cumprod[t])` 并通过 `ReferenceAttentionControl.set_active_sigma(sigma)` 通知 controller。
- 训练 loop 在 `IPGTrainModel.forward` 内根据 `timesteps` 取 batch 平均 sigma 通知 controller，因此训练侧此字段同样会生效（前提是设为非负值）；`alphas_cumprod` 通过 `register_buffer(persistent=False)` 缓存，不入 checkpoint。
- `sigma > timestep_max_sigma` 的步直接跳过颜色 cross-attention 注入。
- 设为 `null` 或负数即关闭该门。训练 yaml 默认 `-1` 是为了让 attn2 在全噪声段都被监督，避免 train/test 推理分布上 over-restrict；要 train/test 严格对齐时，把训练 yaml 也改成 `0.5` 即可。

### 3.1.2 Layer gating（路径 3，训练 + 推理）

仅在中高分辨率（更靠近输出端）的 transformer block 注入颜色，跳过最深的低分辨率层（这些层主要负责整体姿势骨架）。

```yaml
features.color_structure.min_layer_weight: 0.3
```

- 利用既有的 `module.attn_weight ∈ [0, 1]`：`0` 表示通道最多的最深层，`1` 表示通道最少的最浅层。
- `attn_weight < min_layer_weight` 的层完全跳过颜色注入。
- 设为 `0.0` 时退化为"所有层都注入"。

### 3.1.3 Hard query gating（路径 4，训练 + 推理）

把 `target_upper_mask / target_lower_mask` 从"软 mask 后乘"升级成"硬 0/1 写回 mask"，杜绝颜色 residual 通过 soft mask 边缘渐变泄漏到肢体附近。

```yaml
features.color_structure.hard_query_gating: true
features.color_structure.query_threshold: 0.5
```

实现差异（在 `IPG/src/models/mutual_self_attention.py` 颜色注入分支）：

```text
soft (旧)：
    attn_out = attn2(norm_h, K=color_kv)         # 全空间 query 都和颜色 token 算 attn
    cross   += scale * mask_soft * attn_out      # 在外面乘 soft mask（边缘渐变会泄漏）

hard (新)：
    attn_out = attn2(norm_h, K=color_kv)         # attn 计算不变
    gate     = (mask_soft > query_threshold)     # 硬 0/1 写回 mask
    cross   += scale * gate * attn_out           # mask 外区域颜色 residual 严格为 0
```

> 设计上曾尝试把 query 也用 mask 预先置零（`norm_h * gate`），但全零向量经过 attn2 仍会得到 `softmax(0·K)·V = mean(V)` 的非零输出，相当于在 mask 外注入了一个 query 无关的均匀颜色 bias，反而增加了数值噪声且无意义；最终采用的"输入不动 + 输出端硬 gate"是验证后等价但更稳定的实现。

设为 `false` 时退化为旧的 soft 后乘逻辑。

### 3.1.4 推理侧 CLI 覆盖

`Market_gen.py` 新增四个 CLI 选项，用于不改 yaml 快速验证：

```bash
python Market_gen.py \
  --color_timestep_max_sigma 0.4 \
  --color_min_layer_weight 0.5 \
  --color_query_threshold 0.6 \
  --color_hard_gating true
```

`--color_timestep_max_sigma` 传负值即关闭 timestep gate；`--color_hard_gating` 取值 `true` / `false`。

### 3.1.5 与已训练 checkpoint 的兼容性

- 三段门控**不引入任何可学习参数**，老 checkpoint 直接加载即可推理。
- 训练时若希望完全复现旧行为，把 `timestep_max_sigma: -1`、`min_layer_weight: 0.0`、`hard_query_gating: false` 三项都设上即可（也是当前训练 yaml 默认值）。
- `IPGTrainModel` 新增的 `alphas_cumprod` 是 `register_buffer(persistent=False)`，不写入 checkpoint，老的 train state 可以无缝 resume。
- 所有 gating 都是只在 `attn2` 输出端做加权/跳过，不改 `denoising_unet.pth` 的 state_dict shape。

### 3.1.6 调参建议

观察到生成结果"颜色对、但肢体扭曲/手臂多余"时，按下面顺序逐项收紧：

1. 先把 `color_scale` 从 `0.25` 降到 `0.15`，验证肢体改善幅度；
2. 把 `timestep_max_sigma` 从 `0.5` 收到 `0.3`，让颜色只参与最后约 1/3 的步；
3. `min_layer_weight` 从 `0.3` 抬到 `0.5`，让中分辨率层也不再注入颜色，颜色只在最浅 1/2 层细化；
4. `query_threshold` 从 `0.5` 提到 `0.7`，进一步缩小颜色生效区域；
5. 仍不行就要么换更准的 SAM 服装 mask（`features.part_reference_bank.masks.*`），要么彻底关掉颜色注入（`features.color_structure.enabled: false`），让颜色完全由 reference image / part bank 来表达。

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

- `IPG/src/models/mutual_self_attention.py`：三路 reference bank、mask 融合、颜色 cross-attention 注入；新增 timestep / layer / hard query 三段门控（路径 2/3/4），`set_active_sigma` 与 `set_color_gating` 两个 setter。
- `IPG/src/models/color_condition.py`：冻结 CLIP 文本编码器。
- `IPG/src/models/pose_loss.py`：可微近似姿势损失。
- `IPG/src/utils/mask_utils.py`：mask fallback、pose 粗 mask、attention token 对齐。
- `IPG/tools/extract_sam3_part_masks.py`：SAM3 离线提取上/下半身语义 mask。
- `IPG/src/ipg_dataset.py`：返回 mask 与颜色文本。
- `IPG/train_ipg.py`：训练链路、损失、checkpoint 保存；从 yaml 读取并透传颜色门控参数到 `IPGTrainModel.reference_control_reader`，`IPGTrainModel` 缓存 `alphas_cumprod` 以便每个 batch 把平均 sigma 通知 controller，让训练侧 sigma gate 与推理对齐。
- `IPG/src/pipelines/pipeline.py` 和 `IPG/Market_gen.py`：推理链路；`__call__` 与 CLI 都增加四个颜色门控参数，denoising loop 每步通过 `set_active_sigma` 通知 controller 当前噪声级别。

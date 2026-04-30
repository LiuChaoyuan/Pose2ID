# Pose2ID IPG 改进实施计划

## Summary
- 在 IPG 训练和推理链路中加入三个可消融模块：Part-Aware Reference Bank、颜色结构信息注入、可微近似姿势损失。
- 保持旧 checkpoint 和旧数据集可运行：缺少 mask 或颜色 JSON 时走退化路径，任一模块可通过配置关闭。
- 新增一份改进说明文档，写清模块思路、配置开关、数据准备和启动方式。

## Key Changes
- 改造 `E:\codes\Python\Pose2ID\IPG\src\models\mutual_self_attention.py`：
  - 将当前单 `bank` 扩展为 `global / upper / lower` 三类 bank。
  - 写入阶段通过 `active_bank` 分三次写入 reference UNet。
  - 读取阶段用 target upper/lower mask 对三路 attention 输出做残差融合。
  - 新增 `PartBankFusion` 小模块保存可学习 `lambda_upper/lambda_lower`，初始为 `0.0`，单独保存为 `part_bank_fusion.pth`。

- 扩展训练和推理数据流：
  - `E:\codes\Python\Pose2ID\IPG\src\ipg_dataset.py` 返回 `ref_upper_mask/ref_lower_mask/target_upper_mask/target_lower_mask` 与上/下衣颜色文本。
  - mask 优先从配置路径加载；缺失时使用 soft 上下半身退化 mask，target mask 可从 pose 图非零区域加上下半身分区近似生成。
  - `E:\codes\Python\Pose2ID\IPG\src\pipelines\pipeline.py`、`E:\codes\Python\Pose2ID\IPG\Market_gen.py` 支持推理时传入或自动生成这些 mask。

- 新增颜色结构注入：
  - 新建 `E:\codes\Python\Pose2ID\IPG\src\models\color_condition.py`，使用 Stable Diffusion 的 CLIP tokenizer/text encoder 冻结编码颜色文本。
  - 从 `/root/autodl-fs/datasets/market1501/clothing_colors_nl.json` 按参考图文件名读取 `upper/lower` 文本。
  - 在 denoising UNet cross-attention 中分别计算 upper/lower color token 输出，并用 target upper/lower mask 加权注入，默认 `color_scale=0.25`。

- 新增可微近似姿势损失：
  - 新建 `E:\codes\Python\Pose2ID\IPG\src\models\pose_loss.py`。
  - 在低噪声样本上从 `model_pred` 估计 `x0_pred`，经冻结 VAE decode 得到生成图。
  - 用 differentiable edge/soft distance-map 近似 `D(P_gen)`，用 target pose 图生成 `D(P_target)`，计算 MSE。
  - 总损失为 `loss = loss_mse + pose_loss.weight * loss_pose`，默认 `enabled=true`、`weight=0.05`、`max_sigma=0.35`。

## Public Interfaces
- 在 `E:\codes\Python\Pose2ID\IPG\configs\train_ipg.yaml` 新增：
  - `features.part_reference_bank.enabled`
  - `features.color_structure.enabled`
  - `features.pose_loss.enabled`
  - mask roots、颜色 JSON 路径、CLIP 路径、`lambda_init`、`color_scale`、`pose_loss.weight/max_sigma`
- 在 `E:\codes\Python\Pose2ID\IPG\configs\inference.yaml` 新增对应推理开关和路径。
- checkpoint 保存新增 `part_bank_fusion.pth`；旧 checkpoint 缺少该文件时自动用默认初值，不中断推理。
- `Market_gen.py` 新增可选 CLI 参数：`--color_json`、`--mask_root`、`--disable_part_bank`、`--disable_color_structure`。

## Test Plan
- 运行 Python 语法检查：`python -m compileall IPG/src IPG/train_ipg.py IPG/Market_gen.py IPG/inference.py`。
- 用 synthetic tensor 做单元级 smoke test：
  - 三类 bank 写入/同步/清空不报错。
  - mask resize 后形状为 `[B, L, 1]`，CFG batch 下能正确扩展。
  - `lambda_upper/lambda_lower=0` 时输出退化为原 global bank 行为。
- 用临时小数据集测试 `IPGDataset`：
  - 有 mask/color JSON 时能加载。
  - 缺失 mask/color JSON 时 fallback 不报错。
- 若当前机器无 CUDA 或缺少模型权重，只做 CPU 级导入与形状测试；完整训练/推理 smoke test 留到具备权重和 GPU 的环境执行。

## Assumptions
- SAM3 暂未集成到仓库，因此本次只实现“消费 SAM3 mask 结果”的接口与 fallback mask，不内置 SAM3 推理。
- 颜色 JSON 以参考图 basename 作为 key，例如 `0002_c1s1_000451_03.jpg`。
- 姿势损失采用你选择的“可微近似”版本，不在训练内调用外部 DWPose/SAM3 检测器。
- 现有未提交改动会保留，尤其是 `Market_gen.py` 和 `train_ipg.yaml` 里的 `/root/autodl-fs/...` 路径修改。

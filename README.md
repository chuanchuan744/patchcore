# PatchCore 变体实验说明

本仓库包含 4 个脚本，用于在同一数据集上对 PatchCore 做不同组合实验：

- `1.py`：PatchCore 原版
- `1_cosine.py`：PatchCore + cosine 距离
- `1_kmeans.py`：PatchCore + KMeans 后处理
- `1_cosine_kmeans.py`：PatchCore + cosine 距离 + KMeans 后处理

---

## 1. 目录结构

默认使用如下目录（已在代码中固定）：

```text
dataset/
  train/
    good/
  test/
    good/
    color/
    cut/
    hole/
    metal_contamination/
    thread/
  ground_truth/
    color/
    cut/
    hole/
    metal_contamination/
    thread/
```

> 说明：脚本会自动读取 `dataset` 目录并检查上述子目录是否存在。

---

## 2. 四个脚本的区别

### `1.py`（原版）

- 使用 PatchCore 默认距离度量（通常是欧氏距离）。
- 输出 image-level / pixel-level AUROC、AUPR（若当前 anomalib 版本支持）。
- 保存热力图和叠加图。

### `1_cosine.py`（cosine）

- 会尝试以多种参数名启用 `cosine` 距离（兼容不同 anomalib 版本的参数命名差异）。
- 若当前版本不支持，会自动回退到默认距离。

### `1_kmeans.py`（KMeans 后处理）

- 基于 anomaly map 做二类 KMeans 聚类，得到二值缺陷掩膜。
- 对掩膜做开闭运算，减少噪声。
- 在原始 pixel 指标外，额外计算 KMeans 后处理后的 pixel AUROC / pixel AUPR（有 GT 的前提下）。

### `1_cosine_kmeans.py`（cosine + KMeans）

- 同时启用 cosine 距离尝试和 KMeans 后处理。
- 逻辑上是 `1_cosine.py` 与 `1_kmeans.py` 的组合版。

---

## 3. 运行方式

在仓库根目录执行：

```bash
python 1.py
python 1_cosine.py
python 1_kmeans.py
python 1_cosine_kmeans.py
```

---

## 4. 结果输出

脚本会在 `results/` 下输出可视化结果，常见目录包括：

- `results/visualizations_heatmap`
- `results/visualizations_heatmap_cosine`
- `results/visualizations_kmeans`
- `results/visualizations_cosine_kmeans`

常见输出图片：

- 原始 anomaly gray map
- 伪彩 heatmap
- heatmap 与原图叠加图
- （KMeans 脚本）KMeans mask / 轮廓图 / GT 对比图

---

## 5. 依赖建议

建议使用 Python 3.9+，并安装以下核心依赖：

- `torch`
- `anomalib`
- `opencv-python`
- `numpy`
- `Pillow`
- `scikit-learn`（KMeans 相关脚本需要）

可按实际环境自行选择 CUDA / CPU 版本的 PyTorch。

---

## 6. 常见问题

### Q1：为什么代码里使用了 `os._exit(0)`？

用于在部分环境（尤其 Windows + IDE）中强制结束残留线程，减少“脚本看似结束但仍占 CPU”的情况。

### Q2：AUPR 没有输出怎么办？

不同 anomalib 版本里指标类可能叫 `AUPR` 或 `AUPRC`，脚本已做兼容尝试；若两者都不可用，会仅输出 AUROC。

### Q3：KMeans 指标为什么可能显示“未计算”？

当没有可用 GT 掩膜，或像素标签只有单一类别时，无法计算 ROC/AUPR。

---

## 7. 建议的使用顺序

建议按下面顺序对比：

1. 先跑 `1.py` 建立基线；
2. 再跑 `1_cosine.py` 看距离度量变化；
3. 跑 `1_kmeans.py` 看后处理提升；
4. 最后跑 `1_cosine_kmeans.py` 看组合效果。

这样更容易定位性能变化来自哪个改动。

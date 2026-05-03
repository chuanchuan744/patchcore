import os
import sys
import time
import multiprocessing as mp


# ============================================================
# 距离度量设置
# "cosine"：优先尝试余弦距离
# "euclidean"：默认欧氏距离
# ============================================================
DISTANCE_METRIC = "cosine"


def run_patchcore_pipeline():
    # ============================================================
    # 这些环境变量必须尽量在 torch / cv2 / sklearn 等库导入前设置
    # ============================================================
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    import gc
    import warnings

    import cv2
    import numpy as np
    import torch
    from PIL import Image

    from anomalib.data import Folder
    from anomalib.engine import Engine
    from anomalib.models import Patchcore
    from anomalib.metrics import Evaluator, AUROC

    try:
        cv2.setNumThreads(0)
    except Exception:
        pass

    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    try:
        from anomalib.metrics import AUPR
        AUPR_METRIC = AUPR
    except Exception:
        try:
            from anomalib.metrics import AUPRC
            AUPR_METRIC = AUPRC
        except Exception:
            AUPR_METRIC = None

    def count_images(folder):
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
        total = 0

        if os.path.exists(folder):
            for root, _, files in os.walk(folder):
                for f in files:
                    if os.path.splitext(f)[1].lower() in exts:
                        total += 1

        return total

    def tensor_to_numpy(x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu()

            if x.ndim == 4:
                x = x[0]

            if x.ndim == 3 and x.shape[0] in [1, 3]:
                x = x.permute(1, 2, 0)

            x = x.numpy()

        return x

    def normalize_map(anomaly_map):
        anomaly_map = tensor_to_numpy(anomaly_map)
        anomaly_map = np.squeeze(anomaly_map)

        min_val = anomaly_map.min()
        max_val = anomaly_map.max()

        if max_val - min_val < 1e-8:
            return np.zeros_like(anomaly_map, dtype=np.uint8)

        anomaly_map = (anomaly_map - min_val) / (max_val - min_val)
        anomaly_map = (anomaly_map * 255).astype(np.uint8)

        return anomaly_map

    def read_gt_mask(image_path, dataset_root):
        rel_path = os.path.relpath(image_path, dataset_root)
        parts = rel_path.split(os.sep)

        if len(parts) < 3:
            return None

        split_name = parts[0]
        defect_class = parts[1]
        filename = parts[2]

        if split_name != "test" or defect_class == "good":
            return None

        name, _ = os.path.splitext(filename)

        possible_paths = [
            os.path.join(dataset_root, "ground_truth", defect_class, name + "_mask.png"),
            os.path.join(dataset_root, "ground_truth", defect_class, name + ".png"),
            os.path.join(dataset_root, "ground_truth", defect_class, name + "_mask.bmp"),
            os.path.join(dataset_root, "ground_truth", defect_class, name + ".bmp"),
            os.path.join(dataset_root, "ground_truth", defect_class, name + "_mask.jpg"),
            os.path.join(dataset_root, "ground_truth", defect_class, name + ".jpg"),
        ]

        for path in possible_paths:
            if os.path.exists(path):
                gt = Image.open(path).convert("L")
                gt = np.array(gt)
                gt = (gt > 0).astype(np.uint8) * 255
                return gt

        return None

    def save_visualizations(image_path, anomaly_map, save_base_path, dataset_root):
        image = Image.open(image_path).convert("RGB")
        image = np.array(image)

        heatmap = normalize_map(anomaly_map)
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

        heatmap_overlay = cv2.addWeighted(image, 0.6, heatmap_color, 0.4, 0)

        gt_mask = read_gt_mask(image_path, dataset_root)
        compare_overlay = image.copy()

        if gt_mask is not None:
            gt_mask = cv2.resize(
                gt_mask,
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

            gt_color = np.zeros_like(image)
            gt_color[:, :, 1] = gt_mask

            compare_overlay = cv2.addWeighted(image, 0.7, gt_color, 0.3, 0)
            compare_overlay = cv2.addWeighted(compare_overlay, 0.75, heatmap_color, 0.25, 0)

        os.makedirs(os.path.dirname(save_base_path), exist_ok=True)

        Image.fromarray(heatmap).save(save_base_path + "_anomaly_map_gray.png")
        Image.fromarray(heatmap_color).save(save_base_path + "_heatmap_color.png")
        Image.fromarray(heatmap_overlay).save(save_base_path + "_heatmap_overlay.png")

        if gt_mask is not None:
            Image.fromarray(compare_overlay).save(save_base_path + "_compare_gt_heatmap.png")

    def get_batch_value(batch, keys):
        for key in keys:
            if hasattr(batch, key):
                return getattr(batch, key)

            if isinstance(batch, dict) and key in batch:
                return batch[key]

        return None

    def create_patchcore_model(pre_processor, evaluator):
        base_kwargs = dict(
            backbone="wide_resnet50_2",
            layers=["layer2", "layer3"],
            coreset_sampling_ratio=0.01,
            num_neighbors=9,
            pre_processor=pre_processor,
            evaluator=evaluator,
        )

        if DISTANCE_METRIC.lower() in ["cosine", "cos"]:
            cosine_candidate_kwargs = [
                {"distance_metric": "cosine"},
                {"metric": "cosine"},
                {"nearest_neighbor_metric": "cosine"},
            ]

            for extra_kwargs in cosine_candidate_kwargs:
                try:
                    model = Patchcore(
                        **base_kwargs,
                        **extra_kwargs,
                    )
                    print(f"已尝试启用 cosine 距离参数: {extra_kwargs}")
                    return model
                except TypeError:
                    continue

            print(
                "警告: 当前 anomalib 版本的 Patchcore 初始化参数不支持 cosine 距离。"
            )
            print(
                "将自动回退到默认距离。默认通常是 euclidean。"
            )

        model = Patchcore(**base_kwargs)
        print("当前 PatchCore 使用默认距离度量。")

        return model

    def cleanup_resources(*objects):
        print("\n开始清理子进程资源...")

        for obj in objects:
            try:
                del obj
            except Exception:
                pass

        try:
            gc.collect()
        except Exception:
            pass

        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass

            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass

        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        print("子进程资源清理完成。")

    warnings.filterwarnings(
        "ignore",
        message=".*pre_processor.*already saved during checkpointing.*",
    )

    torch.set_float32_matmul_precision("high")

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    dataset_root = os.path.abspath("dataset")

    train_good = os.path.join(dataset_root, "train", "good")
    test_good = os.path.join(dataset_root, "test", "good")
    ground_truth = os.path.join(dataset_root, "ground_truth")

    abnormal_classes = [
        "color",
        "cut",
        "hole",
        "metal_contamination",
        "thread",
    ]

    abnormal_dirs = [f"test/{cls}" for cls in abnormal_classes]

    print("当前工作目录:", os.getcwd())
    print("dataset_root:", dataset_root)
    print("train/good exists:", os.path.exists(train_good))
    print("test/good exists:", os.path.exists(test_good))
    print("ground_truth exists:", os.path.exists(ground_truth))
    print("目标距离度量:", DISTANCE_METRIC)

    if not os.path.exists(train_good):
        raise FileNotFoundError(f"训练正常样本目录不存在: {train_good}")

    if not os.path.exists(test_good):
        raise FileNotFoundError(f"测试正常样本目录不存在: {test_good}")

    if not os.path.exists(ground_truth):
        raise FileNotFoundError(
            f"pixel-level 评估需要 ground_truth mask，但目录不存在: {ground_truth}"
        )

    for cls in abnormal_classes:
        path = os.path.join(dataset_root, "test", cls)
        print(f"test/{cls} exists:", os.path.exists(path))

        if not os.path.exists(path):
            raise FileNotFoundError(f"测试异常样本目录不存在: {path}")

    train_good_count = count_images(train_good)
    test_good_count = count_images(test_good)

    abnormal_counts = {}
    total_abnormal_images = 0

    for cls in abnormal_classes:
        path = os.path.join(dataset_root, "test", cls)
        count = count_images(path)
        abnormal_counts[cls] = count
        total_abnormal_images += count

    total_test_images = test_good_count + total_abnormal_images

    print("train/good 图片数:", train_good_count)
    print("test/good 图片数:", test_good_count)

    for cls, count in abnormal_counts.items():
        print(f"test/{cls} 图片数:", count)

    print("test 异常图片总数:", total_abnormal_images)
    print("test 总图片数:", total_test_images)

    if total_test_images == 0:
        raise RuntimeError("测试集图片数量为 0，请检查 dataset/test 目录。")

    datamodule = Folder(
        name="demo_patchcore",
        root=dataset_root,
        normal_dir="train/good",
        normal_test_dir="test/good",
        abnormal_dir=abnormal_dirs,
        mask_dir="ground_truth",
        train_batch_size=1,
        eval_batch_size=1,
        num_workers=0,
    )

    pre_processor = Patchcore.configure_pre_processor(image_size=(160, 160))

    metrics = [
        AUROC(fields=["pred_score", "gt_label"], prefix="image_"),
        AUROC(fields=["anomaly_map", "gt_mask"], prefix="pixel_"),
    ]

    if AUPR_METRIC is not None:
        metrics.append(
            AUPR_METRIC(fields=["pred_score", "gt_label"], prefix="image_")
        )
        metrics.append(
            AUPR_METRIC(fields=["anomaly_map", "gt_mask"], prefix="pixel_")
        )
    else:
        print("警告: 当前 anomalib 版本没有成功导入 AUPR/AUPRC，只会输出 AUROC。")

    evaluator = Evaluator(test_metrics=metrics)

    model = create_patchcore_model(
        pre_processor=pre_processor,
        evaluator=evaluator,
    )

    engine = Engine(
        default_root_dir="results",
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
    )

    result = None
    predictions = None

    try:
        print("开始建立 memory bank...")
        fit_start = time.perf_counter()
        engine.fit(model=model, datamodule=datamodule)
        fit_end = time.perf_counter()
        print(f"memory bank 构建耗时: {fit_end - fit_start:.3f} s")

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        print("开始测试 image-level / pixel-level 指标...")
        test_start = time.perf_counter()
        result = engine.test(model=model, datamodule=datamodule)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        test_end = time.perf_counter()

        test_time = test_end - test_start
        fps = total_test_images / test_time if test_time > 0 else 0.0

        print("测试结果原始输出:", result)
        print(f"测试总耗时: {test_time:.3f} s")
        print(f"FPS: {fps:.6f}")

        print("开始保存异常热力图...")
        save_vis_dir = os.path.join("results", "visualizations_heatmap_cosine")
        os.makedirs(save_vis_dir, exist_ok=True)

        predictions = engine.predict(model=model, datamodule=datamodule)

        saved_count = 0

        for batch in predictions:
            image_paths = get_batch_value(batch, ["image_path", "image_paths", "path"])
            anomaly_maps = get_batch_value(batch, ["anomaly_map", "anomaly_maps"])

            if image_paths is None or anomaly_maps is None:
                continue

            if isinstance(image_paths, str):
                image_paths = [image_paths]

            if isinstance(anomaly_maps, torch.Tensor):
                if anomaly_maps.ndim == 3:
                    anomaly_maps = anomaly_maps.unsqueeze(0)
                anomaly_maps = list(anomaly_maps)

            for image_path, anomaly_map in zip(image_paths, anomaly_maps):
                rel_path = os.path.relpath(image_path, dataset_root)
                rel_no_ext = os.path.splitext(rel_path)[0]
                save_base_path = os.path.join(save_vis_dir, rel_no_ext)

                save_visualizations(
                    image_path=image_path,
                    anomaly_map=anomaly_map,
                    save_base_path=save_base_path,
                    dataset_root=dataset_root,
                )

                saved_count += 1

        print(f"异常热力图保存完成，共保存 {saved_count} 组")
        print(f"保存目录: {os.path.abspath(save_vis_dir)}")

        if isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict):
            metrics_dict = result[0]

            image_auroc = (
                metrics_dict.get("image_AUROC")
                or metrics_dict.get("image_AUROC_0")
                or metrics_dict.get("AUROC")
            )

            pixel_auroc = (
                metrics_dict.get("pixel_AUROC")
                or metrics_dict.get("pixel_AUROC_0")
            )

            image_aupr = (
                metrics_dict.get("image_AUPR")
                or metrics_dict.get("image_AUPRC")
                or metrics_dict.get("AUPR")
                or metrics_dict.get("AUPRC")
            )

            pixel_aupr = (
                metrics_dict.get("pixel_AUPR")
                or metrics_dict.get("pixel_AUPRC")
            )

            print("\n================ 最终结果 ================")
            print(f"Distance Metric Target : {DISTANCE_METRIC}")
            print(
                f"Image AUROC            : {image_auroc:.6f}"
                if image_auroc is not None
                else "Image AUROC            : 未返回"
            )
            print(
                f"Image AUPR             : {image_aupr:.6f}"
                if image_aupr is not None
                else "Image AUPR             : 未返回"
            )
            print(
                f"Pixel AUROC            : {pixel_auroc:.6f}"
                if pixel_auroc is not None
                else "Pixel AUROC            : 未返回"
            )
            print(
                f"Pixel AUPR             : {pixel_aupr:.6f}"
                if pixel_aupr is not None
                else "Pixel AUPR             : 未返回"
            )
            print(f"FPS                    : {fps:.6f}")
            print("=========================================\n")
        else:
            print("未能解析 result 的结构，请检查 anomalib 返回格式。")

    finally:
        cleanup_resources(
            predictions,
            result,
            model,
            engine,
            datamodule,
        )

        print("子进程任务执行完毕，准备强制退出子进程。")
        sys.stdout.flush()
        sys.stderr.flush()

        os._exit(0)


def main():
    print("启动 PatchCore 子进程...")

    ctx = mp.get_context("spawn")
    process = ctx.Process(target=run_patchcore_pipeline)
    process.start()

    process.join()

    if process.is_alive():
        print("检测到子进程仍未退出，强制终止。")
        process.terminate()
        process.join(timeout=3)

        if process.is_alive():
            process.kill()
            process.join(timeout=3)

    print("PatchCore 子进程已结束。")

    sys.stdout.flush()
    sys.stderr.flush()

    os._exit(0)


if __name__ == "__main__":
    mp.freeze_support()
    main()
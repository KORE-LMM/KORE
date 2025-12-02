import json
import os
from typing import List, Any

# 1) 在这里写入你的文件路径（使用绝对路径）
INPUT_JSON = "/hkfs/work/workspace/scratch/lmu_chd4938-MINED_26/Qwen2.5-VL-null-space/LLaMA-Factory/data/new_10_random_reply_training_data.json"
OUTPUT_JSON = "/hkfs/work/workspace/scratch/lmu_chd4938-MINED_26/Qwen2.5-VL-null-space/LLaMA-Factory/data/2new_10_random_reply_training_data.json"

def all_paths_exist(paths: List[str]) -> bool:
    return all(isinstance(p, str) and os.path.exists(p) for p in paths)

def any_paths_exist(paths: List[str]) -> bool:
    return any(isinstance(p, str) and os.path.exists(p) for p in paths)

def filter_existing(paths: List[str]) -> List[str]:
    return [p for p in paths if isinstance(p, str) and os.path.exists(p)]

def get_items_container(data: Any):
    """
    返回 (items_list, parent, key_if_dict)
    - 如果根是 list，返回 (data, None, None)
    - 如果根是 dict 且含 'data' 且为 list，返回 (data['data'], data, 'data')
    - 否则抛出错误
    """
    if isinstance(data, list):
        return data, None, None
    if isinstance(data, dict):
        # 优先尝试常见的 'data' 容器键
        for k in ("data", "items", "records", "dataset"):
            if k in data and isinstance(data[k], list):
                return data[k], data, k
    raise ValueError("无法识别 JSON 根结构：需要 list，或 dict 中包含 list（如 'data'/'items'/'records'/'dataset'）。")

def main():
    # 读取
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        raw = json.load(f)

    items, parent, parent_key = get_items_container(raw)

    # 收集所有可作为替换源的 images（要求该列表中的所有路径都存在）
    donor_images_lists: List[List[str]] = []
    valid_count = 0
    invalid_count = 0

    for it in items:
        imgs = it.get("images")
        if isinstance(imgs, list) and imgs and all_paths_exist(imgs):
            donor_images_lists.append(imgs)
            valid_count += 1
        else:
            invalid_count += 1

    print(f"总条目: {len(items)} | 可用(全部存在)的 images 源: {len(donor_images_lists)} | 待修复: {invalid_count}")

    fixes = 0
    filtered_used = 0
    emptied = 0

    # 修复阶段
    for it in items:
        imgs = it.get("images")
        if not isinstance(imgs, list):
            imgs = []
            it["images"] = imgs

        if imgs and all_paths_exist(imgs):
            continue  # 已全部存在，无需修改

        # 优先：若存在完整可用的 donor，直接替换
        if donor_images_lists:
            it["images"] = donor_images_lists[0]
            fixes += 1
            continue

        # 其次：没有完整 donor 时，尽量保留本条目中存在的图片
        existing = filter_existing(imgs)
        if existing:
            it["images"] = existing
            filtered_used += 1
        else:
            # 实在没有可用路径，置为空列表
            it["images"] = []
            emptied += 1

    # 写出
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(raw, f, ensure_ascii=False, indent=2)

    print(f"修复完成 -> {OUTPUT_JSON}")
    print(f"- 替换为其他条目可用 images 的条目数: {fixes}")
    print(f"- 仅保留本条目中存在的路径(无完整 donor): {filtered_used}")
    print(f"- 最终为空(无任何存在路径): {emptied}")

if __name__ == "__main__":
    main()
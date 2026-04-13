# split_galaxy10_h5.py
import os
import argparse
import numpy as np
import h5py
from sklearn.model_selection import train_test_split


def copy_subset(src_file, dst_file, indices, compression="gzip"):
    """
    按给定索引，把 src_file 中所有 dataset 子集复制到 dst_file。
    """
    # h5py 花式索引通常要求升序索引，先排序更稳妥
    indices = np.sort(indices)

    with h5py.File(src_file, "r") as fin, h5py.File(dst_file, "w") as fout:
        # 复制全局属性
        for k, v in fin.attrs.items():
            fout.attrs[k] = v

        # 给输出文件写一些说明
        fout.attrs["source_file"] = os.path.abspath(src_file)
        fout.attrs["num_samples"] = len(indices)

        for key in fin.keys():
            obj = fin[key]

            # 这里只处理 dataset；如果以后有 group，可再扩展
            if not isinstance(obj, h5py.Dataset):
                continue

            data_shape = obj.shape
            data_dtype = obj.dtype

            # 只对子样本维度在第 0 维的数据做切分
            # 例如 images: (N, 256, 256, 3), ans: (N,), ra: (N,)
            if len(data_shape) >= 1 and data_shape[0] >= indices.max() + 1:
                new_shape = (len(indices),) + data_shape[1:]
                dset_out = fout.create_dataset(
                    key,
                    shape=new_shape,
                    dtype=data_dtype,
                    compression=compression
                )

                # 分块写入，避免一次性占太多内存
                chunk_size = 512
                for start in range(0, len(indices), chunk_size):
                    end = min(start + chunk_size, len(indices))
                    batch_idx = indices[start:end]
                    dset_out[start:end] = obj[batch_idx]
            else:
                # 如果存在不按样本维切分的 dataset，就原样复制
                fout.create_dataset(key, data=obj[()])


def main():
    parser = argparse.ArgumentParser(description="Split Galaxy10_DECals.h5 into train/test h5 files")
    parser.add_argument("--input", type=str, default="Galaxy10_DECals.h5", help="原始 h5 文件路径")
    parser.add_argument("--test_size", type=float, default=0.2, help="测试集比例，例如 0.2")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--stratify", action="store_true", help="是否按标签分层抽样")
    parser.add_argument("--compression", type=str, default="gzip", help="h5 压缩方式，如 gzip / lzf / None")
    args = parser.parse_args()

    input_path = args.input
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"找不到文件: {input_path}")

    with h5py.File(input_path, "r") as f:
        if "ans" not in f:
            raise KeyError("原始 h5 中未找到 'ans' 标签键")
        labels = np.array(f["ans"])
        n = len(labels)

    indices = np.arange(n)

    stratify_labels = labels if args.stratify else None
    train_idx, test_idx = train_test_split(
        indices,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=stratify_labels
    )

    test_pct = int(round(args.test_size * 100))
    train_pct = 100 - test_pct

    base_dir = os.path.dirname(os.path.abspath(input_path))
    train_file = os.path.join(base_dir, f"Galaxy10_DECals_{train_pct}pct.h5")
    test_file = os.path.join(base_dir, f"Galaxy10_DECals_{test_pct}pct.h5")

    print(f"[INFO] input      : {input_path}")
    print(f"[INFO] total      : {n}")
    print(f"[INFO] train size : {len(train_idx)}")
    print(f"[INFO] test size  : {len(test_idx)}")
    print(f"[INFO] train file : {train_file}")
    print(f"[INFO] test file  : {test_file}")

    copy_subset(input_path, train_file, train_idx, compression=args.compression)
    copy_subset(input_path, test_file, test_idx, compression=args.compression)

    # 给输出文件补充分割信息
    with h5py.File(train_file, "a") as f:
        f.attrs["split"] = "train"
        f.attrs["test_size"] = args.test_size
        f.attrs["random_seed"] = args.seed
        f.attrs["stratify"] = args.stratify

    with h5py.File(test_file, "a") as f:
        f.attrs["split"] = "test"
        f.attrs["test_size"] = args.test_size
        f.attrs["random_seed"] = args.seed
        f.attrs["stratify"] = args.stratify

    print("[DONE] 数据集切分完成。")

    for fn in ["Galaxy10_DECals_80pct.h5", "Galaxy10_DECals_20pct.h5"]:
        with h5py.File(fn, "r") as f:
            print("=" * 50)
            print(fn)
            for k in f.keys():
                print(k, f[k].shape, f[k].dtype)
            print("attrs:", dict(f.attrs))
        
if __name__ == "__main__":
    main()
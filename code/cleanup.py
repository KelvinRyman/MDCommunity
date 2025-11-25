import os
import shutil


def cleanup_pycache(start_path="."):
    pycache_count = 0
    deleted_paths = []

    print(
        f"🚀 开始在路径: '{os.path.abspath(start_path)}' 下搜索并删除 '__pycache__' 文件夹..."
    )

    for dirpath, dirnames, filenames in os.walk(start_path, topdown=False):
        if "__pycache__" in dirnames:
            pycache_path = os.path.join(dirpath, "__pycache__")
            if os.path.isdir(pycache_path):
                try:
                    shutil.rmtree(pycache_path)
                    pycache_count += 1
                    deleted_paths.append(pycache_path)
                    print(f"✅ 已删除: {pycache_path}")
                except Exception as e:
                    print(f"❌ 无法删除 {pycache_path}: {e}")

    print("\n--- 清理完成 ---")
    if pycache_count > 0:
        print(f"🎉 成功删除了 {pycache_count} 个 '__pycache__' 文件夹。")
        # print("删除的路径列表:")
        # for p in deleted_paths:
        #     print(f"  - {p}")
    else:
        print("🔍 在指定路径下没有找到 '__pycache__' 文件夹。")


if __name__ == "__main__":
    cleanup_pycache(".")

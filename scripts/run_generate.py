"""PyInstaller 打包入口（数据生成器 exe）。

用法：pyinstaller 以本文件为入口；打包后双击生成的 exe 即可，
命令行参数与 `python -m generation.generate` 完全一致（参数保持可调）。
"""

import multiprocessing
import os
import sys

# 本地调试 / 打包分析时能找到项目根目录下的包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if __name__ == "__main__":
    multiprocessing.freeze_support()
    from generation.generate import main
    main()

"""支持 `python -m maixrag`，方便未做可编辑安装时直接运行。"""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())

# 项目使用指南

## 使用方法

1. 在终端中 `cd` 到项目路径
2. 修改 `factor_code` 文件夹中的因子文件，每一个因子一个 `.py` 文件，文件名为 `{因子名}.py`
3. 修改 `configs.yaml` 中的因子配置信息
4. 修改 `configs.syspath.py` 中的 `BASE_PATH`，作为项目根目录 (不是必须项，cd到项目路径就不需要了)
5. 安装所需包：

    ```bash
    pip install -r requirements.txt
    ```

6. 运行脚本：

    ```bash
    python main.py
    ```

## 结果显示

1. 因子值会在 `factor_values` 文件夹中
2. 回测信息会在 `backtest` 文件夹，包含：
    1. `stats.csv` 回测的历史记录
    2. `image` 回测绘图
    3. `intermediate` 绘图用到的中间数据

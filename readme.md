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
7. 如果需要配置日更流程， 需要在 `cron_manager.py` 里面配置好需要的时间信息，会自动生成crontab按照指定参数运行，配置好以后
    ```bash
    python cron_manager.py
    ```

## 结果显示

1. 因子值会在 `factor_values` 文件夹中
2. 数据库`trade.backtest_result` 回测的历史记录
2. 回测信息会在 `backtest` 文件夹，包含：
    1. `image` 回测绘图
    2. `intermediate` 绘图用到的中间数据
3. 日志信息在 `logs` 中
3. streamlit 链接: `http://192.168.137.4:8502/`


## 配置清华源

根据需要, 非必须
在终端中
1. 创建 `~/.pip` 目录：

    ```bash
    mkdir -p ~/.pip
    ```

2. 编辑 `pip.conf` 文件：

    ```bash
    nano ~/.pip/pip.conf
    ```

3. 在文件中添加清华源的配置，内容如下：

    ```ini
    [global]
    index-url = https://pypi.tuna.tsinghua.edu.cn/simple
    ```

4. 保存并退出编辑器。对于 `nano` 编辑器，按 `Ctrl + X`，然后按 `Y` 保存，最后按 `Enter` 退出。

5. 现在，Pip 应该能够使用清华源进行安装包了。

    ```bash
    pip install <package_name>
    ```
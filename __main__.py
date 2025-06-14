import argparse
from factorlib.configs.syspath import (WORK_PATH)
import sys

def main():
    parser = argparse.ArgumentParser(description="FactorLib 命令行工具")
    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # 因子计算
    parser_factor = subparsers.add_parser("factor", help="运行因子计算")
    # parser_factor.add_argument("--config", default=WORK_PATH+"/configs/factor.yaml", help="配置文件路径")
    parser_factor.add_argument("--name", default="",help="因子名称")

    # 回测
    parser_backtest = subparsers.add_parser("backtest", help="运行回测")
    # parser_backtest.add_argument("--config", default=WORK_PATH+"/configs/factor.yaml", help="配置文件路径")
    parser_backtest.add_argument("--name", default="",help="因子名称")

    # 绘图
    parser_plot = subparsers.add_parser("plot", help="生成绘图")
    # parser_plot.add_argument("--config", default=WORK_PATH+"/configs/factor.yaml", help="配置文件路径")
    parser_plot.add_argument("--name", default="", help="因子名称")

    # 全量执行
    parser_main = subparsers.add_parser("main", help="全量执行")
    # parser_plot.add_argument("--config", default=WORK_PATH+"/configs/factor.yaml", help="配置文件路径")
    parser_main.add_argument('--name', nargs='+', type=str, required=True,help='输入的数组参数，用空格分隔')

    # 定时任务
    parser_cron = subparsers.add_parser("cron", help="配置定时任务")
    parser_cron.add_argument('--name', nargs='+', type=str, required=True,help='输入的数组参数，用空格分隔')
    parser_cron.add_argument('--time', nargs='+', type=str, required=True, help='输入的数组参数，用空格分隔')

    args = parser.parse_args()

    if args.command == "factor":
        from factorlib import run_factor
        run_factor.main(args.name)
    elif args.command == "backtest":
        from factorlib import run_backtest
        run_backtest.main(args.name)
    elif args.command == "plot":
        from factorlib import run_plot
        run_plot.main(args.name)
    elif args.command == "main":
        from factorlib import main
        s = ",".join(args.name)
        main.tmain(s)
    elif args.command == "cron":
        from factorlib import cron_manager
        s = ",".join(args.name)
        t = ",".join(args.time)
        cron_manager.tmain(s,t)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

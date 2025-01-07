# # main.py
import subprocess
import argparse
import sys
import yaml
import multiprocessing

def run_scripts(factor_name, config_path):
    subprocess.check_call([sys.executable, 'run_factor.py', '--name', factor_name, '--config', config_path])
    subprocess.check_call([sys.executable, 'run_backtest.py', '--name', factor_name, '--config', config_path])
    subprocess.check_call([sys.executable, 'run_plot.py', '--name', factor_name, '--config', config_path])

def main():
    parser = argparse.ArgumentParser(description='Run factor computations, backtest, and plotting for multiple factors.')
    parser.add_argument('--config', type=str, default='/home/yangzhilin/backtest/backtest_light/code/configs/config.yaml', help='Path to the config YAML file.')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    factors = config.get('factors', [])

    processes = []
    for factor in factors:
        p = multiprocessing.Process(target=run_scripts, args=(factor['name'], args.config))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()

if __name__ == "__main__":
    main()


# main.py
# import subprocess
# import argparse
# import sys
# import yaml
# import multiprocessing
# from xalpha import Xalpha

# def run_all_steps(factor_params, config_path):
#     simulator = Xalpha(factor_params)
#     simulator.run()
#     simulator.report_stats()
#     simulator.report_plot(
#         stats={
#             'pot': None,
#             'hd': None,
#             'mdd': None,
#             'wratio': None,
#             'ir': None,
#             'ypnl': None,
#             'sharpe': None,
#             'max_leverage_ratio': None
#         },
#         plot=True,
#         savefig=True,
#         path=None,
#         full_title=f"{simulator.name}_{simulator.fre}"
#     )

# def process_factor(factor_params, config_path):
#     run_all_steps(factor_params, config_path)

# def main():
#     parser = argparse.ArgumentParser(description='Run factor computations, backtest, and plotting for multiple factors.')
#     parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to the config YAML file.')
#     args = parser.parse_args()

#     with open(args.config, 'r') as f:
#         config = yaml.safe_load(f)
#     factors = config.get('factors', [])

#     processes = []
#     for factor in factors:
#         p = multiprocessing.Process(target=process_factor, args=(factor, args.config))
#         p.start()
#         processes.append(p)
    
#     for p in processes:
#         p.join()

# if __name__ == "__main__":
#     main()


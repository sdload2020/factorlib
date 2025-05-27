# run_factor.py
import time
import yaml
import argparse
from factorlib.calc_alpha import AlphaCalc
import os
from pathlib import Path
import ast
from factorlib.configs.syspath import (BASE_PATH,FACTOR_CODE_PATH)

factor_config_path = os.path.join(BASE_PATH, 'configs', 'factor.yaml')
def run_factor(params):
    print("getting params in run_factor")
    simulator = AlphaCalc(params)
    print ("running factor")
    indicator_dict = simulator.run()
    return indicator_dict

def main(factor_name):
    start_time = time.time()

    fileName = factor_name + '.py'
    path = Path(FACTOR_CODE_PATH)
    for file_path in path.rglob('*.py'):  # 使用 rglob 递归匹配所有文件
        if file_path.is_file():
            if(file_path.name == fileName):
                with open(file_path, 'r',encoding='utf-8') as f:
                    source = f.read()
                tree = ast.parse(source)
                arrays = {}

                for node in tree.body:
                    if isinstance(node, ast.Assign):
                        for target in node.targets:
                            if isinstance(target, ast.Name) and target.id == 'config':
                                arrays = ast.literal_eval(node.value)
                                break


    factor_params = next((f for f in arrays['factors'] if f['name'] == factor_name), None)
    print(factor_params)
    if factor_params is None:
        raise ValueError(f"Factor {factor_name} not found in the config file.")
    indicator_dict_all = run_factor(factor_params)
    print(indicator_dict_all)
    
    end_time = time.time()
    total_time = end_time - start_time
    print(f"run_factor.py Total script runtime: {total_time:.2f} seconds")

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run factor computation for a specific factor.')
    parser.add_argument('--name', type=str, required=True, help='Name of the factor to compute.')
    parser.add_argument('--config', type=str, default= factor_config_path, help='Path to the config YAML file.')
    args = parser.parse_args()
    name = args.name
    # fileName = args.name + '.py'
    start_time = time.time()
    with open(args.config, 'r',encoding='utf-8') as f:
        config = yaml.safe_load(f)
    factor_params = next((f for f in config['factors'] if f['name'] == args.name), None)
    if factor_params is None:
        raise ValueError(f"Factor {args.name} not found in the config file.")
    indicator_dict_all = run_factor(factor_params)
    print(indicator_dict_all)
    
    end_time = time.time()
    total_time = end_time - start_time
    print(f"run_factor.py Total script runtime: {total_time:.2f} seconds")
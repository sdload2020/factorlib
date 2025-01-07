# run_factor.py
import time
import yaml
import argparse
from xalpha import Xalpha

def run_factor(params):
    print("getting params in run_factor")
    simulator = Xalpha(params)
    print ("running factor")
    indicator_dict = simulator.run()
    return indicator_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run factor computation for a specific factor.')
    parser.add_argument('--name', type=str, required=True, help='Name of the factor to compute.')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to the config YAML file.')
    args = parser.parse_args()

    start_time = time.time()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    factor_params = next((f for f in config['factors'] if f['name'] == args.name), None)
    if factor_params is None:
        raise ValueError(f"Factor {args.name} not found in the config file.")
    indicator_dict_all = run_factor(factor_params)
    print(indicator_dict_all)
    
    end_time = time.time()
    total_time = end_time - start_time
    print(f"run_factor.py Total script runtime: {total_time:.2f} seconds")

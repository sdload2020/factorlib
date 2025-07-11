import os
from configs.syspath import (
    BASE_PATH, LOGS_PATH, DATA_PATH, UNIVERSE_PATH, FACTOR_VALUES_PATH,
    BACKTEST_PATH, IMAGE_PATH, INTERMEDIATE_PATH, STATS_PATH,RUN_PYTHON_PATH
)
from loguru import logger
from utils.logger_setup import setup_execution_logger
setup_execution_logger(LOGS_PATH)

def generate_crontab_entry(
    minute='57',
    hour='14',
    day_of_month='*',
    month='*',
    day_of_week='*',
    python_path='/home/yangzhilin/anaconda3/envs/yangzl39/bin/python',
    script_path=BASE_PATH,
    script='main.py'
    # log_file=os.path.join(LOGS_PATH, 'cron.log')
):

    command = (
        f'cd {script_path} && '
        f'{python_path} {script}'
    )
    cron_time = f'{minute} {hour} {day_of_month} {month} {day_of_week}'
    cron_entry = f'{cron_time} {command}'
    return cron_entry

def generate_crontab_entry2(
    minute='57',
    hour='14',
    day_of_month='*',
    month='*',
    day_of_week='*',
    python_path='/home/yangzhilin/anaconda3/envs/yangzl39/bin/python',
    script_path=BASE_PATH,
    script='main.py'
    # log_file=os.path.join(LOGS_PATH, 'cron.log')
):

    command = (
        # f'cd {script_path} && '
        f'{python_path} {script}'
    )
    cron_time = f'{minute} {hour} {day_of_month} {month} {day_of_week}'
    cron_entry = f'{cron_time} {command}'
    return cron_entry

def add_crontab_entry(cron_entry):
    os.system('crontab -l > mycron 2>/dev/null')
    with open('mycron', 'a') as cron_file:
        cron_file.write(cron_entry + '\n')
    os.system('crontab mycron')
    os.remove('mycron')

def delete_all_crontab():
    """
    删除当前用户的所有 crontab 条目。
    """
    os.system('crontab -r')
    logger.info("所有 crontab 条目已被删除。")

def tmain(name,times):
    time = times.split(",")
    logger.info(times)
    minute = time[0]
    hour = time[1]
    day_of_month = time[2]
    month = time[3]
    day_of_week = time[4]

    cron_entry = generate_crontab_entry2(
        minute=minute,
        hour=hour,
        day_of_month=day_of_month,
        month=month,
        day_of_week=day_of_week,  # 一周7天
        python_path=RUN_PYTHON_PATH,  # 替换Python 路径(terminal中输入which python)
        ## 以下不需要修改
        script_path=BASE_PATH,
        script='main --name '+name,
        # log_file=os.path.join(LOGS_PATH, 'cron.log')
    )
    add_crontab_entry(cron_entry)
    logger.info("Crontab 已添加：")
    logger.info(cron_entry)
    return

def main():
    
    cron_entry = generate_crontab_entry(
        minute='07',
        hour='15',
        day_of_month='*',
        month='*',
        day_of_week='*',  # 一周7天
        python_path='/home/yangzhilin/anaconda3/envs/yangzl39/bin/python',  # 替换Python 路径(terminal中输入which python)
        ## 以下不需要修改
        script_path=BASE_PATH,
        script='main.py'
        # log_file=os.path.join(LOGS_PATH, 'cron.log') 
    )
    add_crontab_entry(cron_entry)
    logger.info("Crontab 已添加：")
    logger.info(cron_entry)
    return
    
if __name__ == '__main__':
    # 自定义参数
    cron_entry = generate_crontab_entry(
        minute='07',
        hour='15',
        day_of_month='*',
        month='*',
        day_of_week='*',  # 一周7天
        python_path='/home/yangzhilin/anaconda3/envs/yangzl39/bin/python',  # 替换Python 路径(terminal中输入which python)
        ## 以下不需要修改
        script_path=BASE_PATH,
        script='main.py'
        # log_file=os.path.join(LOGS_PATH, 'cron.log') 
    )
    add_crontab_entry(cron_entry)
    logger.info("Crontab 已添加：")
    logger.info(cron_entry)
    # ============================================================
    # 如果需要删除所有 crontab 条目，可以取消下面的注释
    # delete_all_crontab()
    # logger.info("Crontab 已删除。")


# configs/tablecreator.py

create_backtest_result_table = """
CREATE TABLE IF NOT EXISTS backtest_result (
    id INT AUTO_INCREMENT PRIMARY KEY,
    datetime DATETIME NOT NULL,
    name VARCHAR(255) NOT NULL,
    level VARCHAR(255),
    update_time DATETIME,
    factortype VARCHAR(255),
    if_prod BOOLEAN,
    start_date DATE,
    end_date DATE,
    frequency VARCHAR(50),
    pot FLOAT,
    hd FLOAT,
    mdd FLOAT,
    wratio FLOAT,
    ir FLOAT,
    ypnl FLOAT,
    sharpe FLOAT,
    max_leverage_ratio FLOAT,
    if_crontab BOOLEAN,
    out_sample_date DATE,
    factor_value_path VARCHAR(500),
    factor_code_path VARCHAR(500),
    intermediate_path VARCHAR(500),
    UNIQUE KEY unique_name (name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""

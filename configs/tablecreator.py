# configs/tablecreator.py


create_backtest_result_table = """
CREATE TABLE IF NOT EXISTS backtest_result (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    frequency VARCHAR(50),
    updatetime DATETIME,
    factortype VARCHAR(255),
    factortype2 VARCHAR(255),
    level VARCHAR(255),
    if_prod BOOLEAN,
    start_date DATE,
    end_date DATE,
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
    author VARCHAR(255) NOT NULL,  -- 添加 author 字段
    factor_value_path VARCHAR(500),
    factor_code_path VARCHAR(500),
    intermediate_path VARCHAR(500),
    UNIQUE KEY unique_name_author_frequency (name, author, frequency)  -- 修改唯一键约束
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""
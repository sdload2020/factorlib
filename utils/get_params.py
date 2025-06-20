import ast
import sys
from pathlib import Path
from configs.syspath import FACTOR_CODE_PATH

# 参数传递logger
def get_factor_params(factor_name, logger):
    """
    获取指定因子的配置参数
    
    参数:
        factor_name: 因子名称
        logger: 已配置的 logger 实例
        
    异常:
        如果找不到因子配置，记录错误日志并退出程序
    """
    fileName = factor_name + '.py'
    path = Path(FACTOR_CODE_PATH)
    config_found = False
    
    # 递归搜索因子文件
    for file_path in path.rglob('*.py'):
        if file_path.is_file() and file_path.name == fileName:
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()
            tree = ast.parse(source)
            
            # 解析查找配置
            for node in tree.body:
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id == 'config':
                            arrays = ast.literal_eval(node.value)
                            config_found = True
                            
                            # 在配置array中查找指定因子
                            factor_params = next(
                                (f for f in arrays['factors'] if f['name'] == factor_name), 
                                None
                            )
                            
                            if factor_params is None:
                                logger.error(f"错误: 在 {file_path} 中找到配置文件，但未找到因子 '{factor_name}' 的配置")
                                sys.exit(1)
                                
                            return factor_params
    
    # 如果未找到配置文件
    if not config_found:
        logger.error(f"错误: 未找到因子 '{factor_name}' 的配置文件 ({fileName})")
        sys.exit(1)
    
    logger.error(f"错误: 未知错误，未能获取因子 '{factor_name}' 的配置")
    sys.exit(1)
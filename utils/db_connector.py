import mysql.connector
from mysql.connector import errorcode
from configs.syspath import BASE_PATH
from configs.dbconfig import db_config

def fetch_latest_stats_from_db(factor_name):
    """
    从数据库中获取指定 factor_name 的最新记录
    """
    try:
        # 连接到数据库
        cnx = mysql.connector.connect(**db_config)
        cursor = cnx.cursor(dictionary=True)  # 使用 dictionary=True 返回字典格式的结果

        # 执行查询
        query = """
        SELECT * FROM backtest_result 
        WHERE name = %s 
        ORDER BY updatetime DESC 
        LIMIT 1
        """
        cursor.execute(query, (factor_name,))
        result = cursor.fetchone()

        if not result:
            raise ValueError(f"未找到因子 '{factor_name}' 的记录")

        return result

    except mysql.connector.Error as err:
        # 处理数据库错误
        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            print("错误: 用户名或密码错误。")
        elif err.errno == errorcode.ER_BAD_DB_ERROR:
            print("错误: 数据库不存在。")
        else:
            print(f"发生错误: {err}")
        return None

    finally:
        # 关闭连接
        if 'cursor' in locals():
            cursor.close()
        if 'cnx' in locals() and cnx.is_connected():
            cnx.close()


# delete table function
def delete_table(table_name):
    """
    删除指定表
    """
    try:
        # 连接到数据库
        cnx = mysql.connector.connect(**db_config)
        cursor = cnx.cursor()

        # 执行删除
        query = f"DROP TABLE IF EXISTS {table_name}"
        cursor.execute(query)
        cnx.commit()

    except mysql.connector.Error as err:
        # 处理数据库错误
        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            print("错误: 用户名或密码错误。")
        elif err.errno == errorcode.ER_BAD_DB_ERROR:
            print("错误: 数据库不存在。")
        else:
            print(f"发生错误: {err}")

    finally:
        # 关闭连接
        if 'cursor' in locals():
            cursor.close()
        if 'cnx' in locals() and cnx.is_connected():
            cnx.close()


if __name__ == "__main__":
    # 测试 fetch_latest_stats_from_db
    # factor_name = 'backtest_result'
    # latest_stats = fetch_latest_stats_from_db(factor_name)
    # print(latest_stats)

    # 测试 delete_table
    table_name = 'backtest_result'
    delete_table(table_name)
    print(f"已删除表: {table_name}")
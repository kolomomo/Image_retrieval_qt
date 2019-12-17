import pymysql


class Mysql:
    def __init__(self):
        self.content = pymysql.Connect(
            host='152.136.109.167',  # mysql的主机ip
            port=3306,  # 端口
            user='wb',  # 用户名
            passwd='wewe1011',  # 数据库密码
            db='qtuser',  # 数据库名
            charset='utf8',  # 字符集
        )
        self.cursor = self.content.cursor()

    def query(self):
        sql = "select U_name,U_passwd from user_table;"
        self.cursor.execute(sql)
        for row in self.cursor.fetchall():
            print("U_name:%s\t U_passwd:%s" % row)
        print(f"一共查找到：{self.cursor.rowcount}")

    def end(self):
        self.cursor.close()
        self.content.close()


if __name__ == '__main__':
    mysql = Mysql()
    mysql.query()
    mysql.end()
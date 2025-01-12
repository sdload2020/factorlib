from dingtalkchatbot.chatbot import DingtalkChatbot, ActionCard, CardItem
# WebHook地址
webhook = 'https://oapi.dingtalk.com/robot/send?access_token=这里填写自己钉钉群自定义机器人的token'
secret = 'SEC11b9...这里填写自己的加密设置密钥'  # 可选：创建机器人勾选“加签”选项时使用
# 初始化机器人小丁
xiaoding = DingtalkChatbot(webhook)  # 方式一：通常初始化方式
xiaoding = DingtalkChatbot(webhook, secret=secret)  # 方式二：勾选“加签”选项时使用（v1.5以上新功能）
xiaoding = DingtalkChatbot(webhook, pc_slide=True)  # 方式三：设置消息链接在PC端侧边栏打开（v1.5以上新功能）
# Text消息@所有人
xiaoding.send_text(msg='我就是小丁，小丁就是我！', is_at_all=True)
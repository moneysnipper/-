import streamlit as st
import streamlit_authenticator as stauth
import app

#
# # Check if 'key' already exists in session_state
# # If not, then initialize it
# if 'authentication_status' not in st.session_state:
#     st.session_state['authentication_status'] = 'None'

# 如下代码数据，可以来自数据库
names = ['zdm', '管理员']
usernames = ['zdm', 'admin']
passwords = ['12345', 'ad123']

credentials = {"usernames":{}}

for un, name, pw in zip(usernames, names, passwords):
    user_dict = {"name":name,"password":pw}
    credentials["usernames"].update({un:user_dict})
# 初始化认证器
authenticator = stauth.Authenticate(credentials, "app_home", "auth", cookie_expiry_days=30)

# 登录并获取状态
name, authentication_status, username = authenticator.login("Login", "main")

# Check if 'key' already exists in session_state
# If not, then initialize it
if 'authentication_status' not in st.session_state:
    st.session_state['authentication_status'] = 'None'
# 打印状态信息，帮助调试
print("Name:", name)
print("Authentication status:", authentication_status)
print("Username:", username)
# if authentication_status:
#     # 如果登录成功，显示主页面
#     st.write("Welcome to the main page!")
# else:
#     # 如果登录失败，显示登录页面或错误消息
#     st.write("Login failed. Please check your username and password.")

if authentication_status:
    with st.container():
        cols1,cols2 = st.columns(2)
        cols1.write('欢迎 *%s*' % (name))
        with cols2.container():
            authenticator.logout('Logout', 'main')

    app.main()
elif authentication_status == False:
    st.error('Username/password is incorrect')
elif authentication_status == None:
    st.warning('Please enter your username and password')

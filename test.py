import streamlit as st
import streamlit_authenticator as stauth
import app



if __name__ == "__main__":
    # 用户信息，后续可以来自DB
    names = ['Oil领域用户', '管理员']  # 用户名
    usernames = ['zdm', 'dataManagerAdmin']  # 登录名
    passwords = ['12345', 'Abcd1234!#!']  # 登录密码
    # 对密码进行加密操作，后续将这个存放在credentials中
    hashed_passwords = stauth.Hasher(passwords).generate()

    # 定义字典，初始化字典
    credentials = {'usernames': {}}
    # 生成服务器端的用户身份凭证信息
    for i in range(0, len(names)):
        credentials['usernames'][usernames[i]] = {'name': names[i], 'password': hashed_passwords[i]}
    authenticator = stauth.Authenticate(credentials, 'some_cookie_name', 'some_signature_key', cookie_expiry_days=0)
    name, authentication_status, username = authenticator.login('Login', 'main')

    if authentication_status:  # 登录成功
        app.main()
    elif authentication_status == False:  # 登录失败
        st.error('Username/password is incorrect')
    elif authentication_status == None:  # 未输入登录信息
        st.warning('Please enter your username and password')
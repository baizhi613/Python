from django.shortcuts import render
from django.http import HttpResponse
from django.contrib.auth.models import User
from django.contrib.auth import authenticate,login
def user_reg(request):
    if request.method=="GET":
        return render(request,'6/user_reg.html')
    if request.method=="POST":
        uname=request.POST.get("username",'')
        pwd=request.POST.get("password",'')
        if User.objects.filter(username=uname):
            info="用户名已存在"
        else:
            d=dict(username=uname,password=pwd,email='111@111.com',is_staff=1,is_active=1,is_superuser=1)
            user=User.objects.create_user(**d)
            info="注册成功,请登录"
        return render(request,'6/user_reg.html',{'info':info})
    
def user_login(request):
    if request.method=="GET":
        return render(request,'6/user_login.html')
    if request.method=="POST":
        uname=request.POST.get("username",'')
        pwd=request.POST.gry("password",'')
        if User.objects.filter(username=uname):
            user=authenticate(username=uname,password=pwd)
            if user:
                if user.is_active:
                    login(request,user)
                    info="登录成功"
                else:
                    info="用户未激活"
            else:
                   info="账号密码不对，请重新输入"
        else:
                info="用户账号不存在，请查询"
        return render(request,'6/user_login.html',{'info':info})

                     
                    

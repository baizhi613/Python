import os
from .forms import *
from http.client import HTTPResponse
from django.shortcuts import render
from .models import *
def upload_file(request):
    if request.method == 'GET':
        return render(request,"5/upload.html")
    if request.method == 'POST':
        myfile=request.FILES.get('myfile',None)
        if myfile:
            path='media/uploads/'
        if not os.path.exists(path):
            os.makedirs(path)
        dest=open(os.path.join(path+myfile.name),'wb+')
        for chunk in myfile.chunks():
            dest.write(chunk)
            dest.close()
        return HTTPResponse('上传完成！')
    else:
        return HTTPResponse('没有上传文件！')

def userinfo_form(request):
    if request.method == 'GET':
        myform=UserInfoForm()
        return render(request,"5/userinfo.html",{'form_obj':myform})
    
def userinfo_msg_form(request):
    if request.method == 'POST':
        myform=UserInfo_Msg_Form()
        return render(request,"5/userinfo_msg.html",{'form_obj':myform})
    else:
        f=UserInfo_Msg_Form(request.POST)
        if f.is_valid():
            print(f.clean())
            print(f.cleaned_data['username'])
            print(f.data)
        else:
            errors=f.errors
            print(errors)
            return render(request,"5/userinfoform.html",{'form_obj':f,'errors':errors})
        return render(request,"5/userinfoform.html",{'form_obj':f})
    
def imgfileform(request):
    if request.method=='GET':
        f=ImgFileForm()
        return render(request,"5/upload_form.html",{'form_obj':f})
    else:
        f=ImgFileForm(request.POST,request.FILES)
        if f.is_valid():
            name=f.cleaned_data['name']
            headimg=f.cleaned_data['headimg']
            userimg=ImgFile()
            userimg.name=name
            userimg.headimg=headimg
            userimg.save()
            print("上传成功")
            return render(request,"5/upload_form.html",{'form_obj':f,'user':userimg})

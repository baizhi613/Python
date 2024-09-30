from django.shortcuts import redirect, render,HttpResponse
from django import forms
from django.core.validators import RegexValidator
from utils.helper import check_code
from io import BytesIO
from utils.encrypt import md5
from app import models

class LoginForm(forms.Form):
    username=forms.CharField(
        label="用户名",
        widget=forms.TextInput(attrs={'class':'form-control', 'placeholder':'请输入用户名'}),
    )
    password=forms.CharField(
        label="密码",
        widget=forms.PasswordInput(attrs={'class':'form-control', 'placeholder':'请输入密码'},render_value=True),
    )
    code=forms.CharField(
        label="验证码",
        widget=forms.TextInput(attrs={'class':'form-control', 'placeholder':'输入验证码'}),
    )

def login(request):
    if request.method=='GET':
        form=LoginForm()
        return render(request, 'login.html',{'form':form})
    if request.method=='POST':
        form=LoginForm(data=request.POST)
        if not form.is_valid():
            return render(request, 'login.html',{'form':form})
        
    image_code=request.session.get("image_code")
    if not image_code:
        form.add_error("code","验证码已失效")
        return render(request, 'login.html',{'form':form})
    if image_code.upper()!=form.cleaned_data["code"].upper():
        form.add_error("code","验证码错误")
        return render(request, 'login.html',{'form':form}) 

    user=form.cleaned_data["username"]
    pwd=form.cleaned_data["password"]

    encrypt_password=md5(pwd)
    print(user,encrypt_password)
    admin_object=models.Admin.objects.filter(username=user,password=encrypt_password).first()   
    if not admin_object:
        return render(request, 'login.html',{'form':form,'error':'用户名或密码错误'})

    request.session['info']={"id":admin_object.id,'name':admin_object.username}
    request.session.set_expiry(60*60*24*7)

    return redirect("/home/")

def img_code(request):
    image_object,code_str=check_code()
    stream=BytesIO()
    image_object.save(stream,'png')
    request.session['image_code']=code_str
    request.session.set_expiry(60)
    return HttpResponse(stream.getvalue())

def home(request):
    #request.info_dict['name']
    return render(request, 'home.html')

def logout(request):
    request.session.clear()
    return redirect("/login/")
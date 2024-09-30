import re
from django import forms
from django.core.exceptions import ValidationError

def moblie_validate(value):
    moblie_re=re.compile(r'^(13[0-9]|15[012356789]|17[678]|18[0-9]|14[57])[0-9]{8}')
    if not moblie_re.match(value):
        raise ValidationError('手机号码格式错误')
def age_validate(value):
    if value<1|value>120:
        raise ValidationError('年龄必须在1-120之间')

class UserForm(forms.Form):
    STATUS=((None,'请选择'),(0,'正常'),(1,'无效'),)
    username=forms.CharField(label="用户名称",min_length=6,widget=forms.widgets.TextInput(attrs={'class':'form-control','placeholder':'请输入用户名'}))
    password=forms.CharField(label="密码",min_length=6,max_length=10,widget=forms.widgets.PasswordInput(attrs={"class":"password"},render_value=True))
    age=forms.IntegerField(label="年龄",initial=1)
    moblile=forms.CharField(label="手机号码")
    status=forms.ChoiceField(label="用户状态",choices=STATUS)
    createdate=forms.DateTimeField(label="创建时间",required=False)

class UserInfoForm(forms.Form):
    STATUS=((None,'请选择'),(0,'正常'),(1,'无效'),)
    username=forms.CharField(label="用户名称",min_length=6,widget=forms.widgets.TextInput(attrs={'class':'form-control','placeholder':'请输入用户名'}))
    password=forms.CharField(label="密码",min_length=6,max_length=10,widget=forms.widgets.PasswordInput(attrs={"class":"password"},render_value=True))
    age=forms.IntegerField(label="年龄",initial=1)
    moblile=forms.CharField(label="手机号码")
    status=forms.ChoiceField(label="用户状态",choices=STATUS)
    #createdate=forms.DateTimeField(label="创建时间",required=False)

class UserInfo_Msg_Form(forms.Form):
    STATUS=((None,'请选择'),(0,'正常'),(1,'无效'),)
    username=forms.CharField(label="用户名称",min_length=6,widget=forms.widgets.TextInput(attrs={'class':'form-control','placeholder':'请输入用户名'}),error_messages={'required':'用户名不能为空','min_length':'用户名长度不能小于6','invalid':'不能含有特殊字符'})
    password=forms.CharField(label="密码",min_length=6,max_length=10,widget=forms.widgets.PasswordInput(attrs={"class":"password"},render_value=True))
    age=forms.IntegerField(label="年龄",initial=1,validators=[age_validate],error_messages={'required':'年龄不能为空',})
    mobile=forms.CharField(label="手机号码",validators=[moblie_validate],error_messages={'required':'手机号码不能为空',})
    status=forms.ChoiceField(label="用户状态",choices=STATUS,error_messages={'required':'用户状态不能为空',})
    createdate=forms.DateTimeField(label="创建时间",required=False)

class ImgFileForm(forms.Form):
    name=forms.CharField()
    headimg=forms.ImageField()

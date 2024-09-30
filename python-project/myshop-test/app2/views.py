from django.shortcuts import render
from django.http import HttpResponse
from django.urls import reverse
from django.shortcuts import redirect
from app2.models import UserBaseInfo

def index(request):
    return HttpResponse("app2中的index方法")
def show(request, id):
    return HttpResponse("app2中的show方法，id为"+str(id))
def show_uuid(request, id):
    return HttpResponse("app2中的show_uuid方法，id为"+str(id))
def show_slug(request, q):
    return HttpResponse("app2中的show_slug方法，q为"+str(q))
def article_list(request,year):
    return HttpResponse("app2中的article_list方法，参数为year，指定4位，值为"+str(year))
def article_page(request,page,key):
    return HttpResponse("app2中的article_page方法，参数为page，page值为"+str(page)+"，key值为"+str(key))
def url_reverse(request):
    print("在views()函数中使用reverse()方法解析的结果:"+reverse("app2_url_reverse"))
    return render(request, "2/url_reverse.html")
def hello(request):
    return HttpResponse("Hello Django!!!")

def test_get(request):
    print(request.get_host())#域名+端口
    print(request.get_raw_uri())#完整的url路径
    print(request.path)#获取访问文件路径，不含参数
    print(request.get_full_path())#获取访问文件路径，含参数
    print(request.method)#获取请求方法，如GET、POST等
    print(request.GET)#获取GET请求参数，字典类型
    print(request.META["HTTP_USER_AGENT"])#用户浏览器的user-agent信息
    print(request.META["REMOTE_ADDR"])#获取客户端IP地址
    print(request.GET.get('username'))#获取get参数
    return HttpResponse("")

def test_post(request):
    print(request.method)#获取请求方法，如GET、POST等
    print(request.POST.get('username'))
    return render(request, "2/test_post.html")

def test_response(request):
    response=HttpResponse()
    response.write("Hello Django!!!")
    response.write("<br>")
    response.write(response.content)
    response.write("<br>")
    response.write(response['Content-Type'])
    response.write("<br>")
    response.write(response.status_code)
    response.write("<br>")
    response.write(response.charset)
    response.write("<br>")
    return response

def test_render(request):
    return render(request,'2/test_render.html',{'info':'hello django'},content_type='text/html')

def test_redirect_model(request,id):
    user=UserBaseInfo.objects.get(id=id)
    return redirect(user)
def userinfo(request,id):
    user=UserBaseInfo.objects.get(id=id)
    return HttpResponse("编号:"+str(user.id)+"姓名:"+user.username)
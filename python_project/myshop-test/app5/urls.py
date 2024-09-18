from http import server
from django.conf import settings
from django.urls import path, re_path
from django.views.static import serve
from app5 import views
urlpatterns = [
    path('upload_file/',views.upload_file),
    path('userinfo_form/',views.userinfo_form),
    path('userinfo_msg_form/',views.userinfo_msg_form),
    path('userimg/',views.imgfileform),
    re_path(r'media/(?P<path>.*)',serve,{'document_root':settings.MEDIA_ROOT}),
]

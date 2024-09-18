from django.urls import path,re_path
from app2 import views
urlpatterns = [
    path('app2/index',views.index),
    path('app2/show/<int:id>/',views.show),
    path('app2/article/<uuid:id>/',views.show_uuid,name='show_uuid'),
    path('app2/article/<slug:q>/',views.show_slug,name='show_slug'),
    re_path(r'app2/list/(?P<year>\d{4})/',views.article_list),
    re_path(r'app2/page/(?P<page>\d+)&key=(?P<key>\w+)',views.article_page,name='article_page'),
    path('app2/url_reverse/',views.url_reverse,name='app2_url_reverse'),

    path('app2/test_get/',views.test_get),
    path('app2/test_post/',views.test_post),
    path('app2/test_response/',views.test_response),
    path('app2/test_redirect_model/',views.test_redirect_model,name='app2_test_redirect_model'),
    path('app2/userinfo/<int:id>/',views.userinfo,name='app2_userinfo'),
]
from django.urls import path
from app3 import views
urlpatterns = [
    path('var/',views.var),
    path('for_lable/',views.for_lable),
    path('filter/',views.filter),
    path('html_filter/',views.html_filter),
    path('diy_filter/',views.diy_filter),
    path('diy_tags/',views.diy_tags),
    path('show_info/',views.show_info),
    path('welcome/',views.welcome),
    path('base_include/',views.base_include),
]

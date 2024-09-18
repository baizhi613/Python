from django.shortcuts import redirect
from django.utils.deprecation import MiddlewareMixin

class AuthMiddleware(MiddlewareMixin):
    def process_request(self, request):
        if request.path_info in ["/login/","/img/code/"]:
            return
        
        info_dict=request.session.get('info')
        if not info_dict:
            return redirect('/login/') 
        
        request.info_dict=info_dict
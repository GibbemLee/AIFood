from django.db import IntegrityError
from django.shortcuts import render
from django.http import HttpResponseRedirect, HttpResponse
from django.contrib.auth.hashers import check_password
from django.contrib.auth import get_user_model
from django.contrib import auth
from django.views.generic import View, FormView, ListView
from .forms import UserForm, SignInForm
from info.models import Book

def main(request):
    if request.COOKIES.get('username') is not None:
        ID = request.COOKIES.get('username')
        pw = request.COOKIES.get('password')
        user = auth.authenticate(request, username=ID, password=pw)
        
        if user is not None:
            auth.login(request, user)
            return render(request, 'main/main.html')
    
    return render(request, 'main/main.html')

class SignupView(FormView):
    form_class = UserForm
    template_name = 'main/signup.html'
    success_url = '/main/signup/done/'
    
    def form_valid(self, form):
        
        ID = form.cleaned_data['user_id']
        email = form.cleaned_data['email']
        pw = form.cleaned_data['pw']
        pw2 = form.cleaned_data['pw2']
        try:
            if pw == pw2:
                User = get_user_model()
                new_user = User.objects.create_user(username=ID, email=email, password=pw)
                auth.login(self.request, new_user)
                    
                return super(SignupView, self).form_valid(form)
            else:
                form = UserForm()
                return render(self.request, self.template_name, {'form':form, 'error_message': 'password is incorrect'})
        except IntegrityError:
            form = UserForm()
            return render(self.request, self.template_name, {'form':form, 'error_message': 'same user does already exist'})
    
class LoginView(FormView):
    form_class = SignInForm
    template_name = 'main/login.html'
    success_url = '/main/'
    
    def form_valid(self, form):   
        if self.request.COOKIES.get('username') is not None:
            ID = self.request.COOKIES.get('username')
            pw = self.request.COOKIES.get('password')
            user = auth.authenticate(self.request, username=ID, password=pw)
            
            if user is not None:
                auth.login(self.request, user)
                return HttpResponseRedirect("/main/")
            else:
                form = SignInForm
                return render(self.request, self.template_name, {'form':form, 'error_message':'username or password is incorrect'})

                
        else:
            ID = form.cleaned_data['user_id']
            pw = form.cleaned_data['pw']
            user = auth.authenticate(self.request, username=ID, password=pw)
            
            if user is not None:
                auth.login(self.request, user)
                response = HttpResponseRedirect('/main/')
                response.set_cookie('username', ID)
                response.set_cookie('password', pw)
                return response
        
            else:
                form = SignInForm
                return render(self.request, self.template_name, {'form':form, 'error_message':'username or password is incorrect'})

def logout(request):
    response = HttpResponseRedirect('/main/')
    response.delete_cookie('username')
    response.delete_cookie('password')
    auth.logout(request)
    return response


def mypage(request):
    return render(request, 'main/mypage.html')

def deleteUser(request):
    context = {}
    
    if request.method == "POST":
        ID = request.POST.get("user")
        pw = request.POST.get("pw")
        user = auth.authenticate(request, username=ID, password=pw)
        
        if user is not None:
            user.delete()
            return HttpResponse('<script type="text/javascript">opener.location.href = "/main/"; self.close();</script>')
        else:
            context.update({'error':"password is incorrect"})
            return render(request, 'main/deleteUser.html', context)
            
        
    return  render(request, 'main/deleteUser.html')

def changePW(request):
    context = {}
    if request.method == "POST":
        current_password = request.POST.get("origin_pw")
        user = request.user
        if check_password(current_password,user.password):
            new_password = request.POST.get("pw1")
            password_confirm = request.POST.get("pw2")
            if new_password == password_confirm:
                user.set_password(new_password)
                user.save()
                auth.login(request,user)
                return HttpResponse('<script type="text/javascript">window.close();</script>')
            else:
                context.update({'error':"새로운 비밀번호를 다시 확인해주세요."})
        else:
            context.update({'error':"현재 비밀번호가 일치하지 않습니다."})
    return render(request, 'main/changepw.html', context)

class RatingView(ListView):
    template_name = 'main/rating.html'
    success_url = '/info/'
    model = Book
    paginate_by = 30
    queryset = Book.objects.all()
    
    def post(self, request):
        if request.method == "POST":
            username = request.POST.get("user")
            
            return HttpResponseRedirect("/info/")



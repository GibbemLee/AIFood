from django.urls import path
from . import views
from django.views.generic import TemplateView

app_name = 'main'
urlpatterns = [
    path('', views.main, name='main'),
    path('rating/', views.RatingView.as_view(), name='rating'),
    path('login/', views.LoginView.as_view(), name='login'),
    path('logout/', views.logout, name='logout'),
    path('mypage/', views.mypage, name='mypage'),
    path('mypage/changepw/', views.changePW, name='changepw'),
    path('mypage/deleteUser/', views.deleteUser, name='deleteUser'),
    path('signup/', views.SignupView.as_view(), name='signup'),
    path('signup/done/', TemplateView.as_view(template_name="main/signup_done.html")),
    ]
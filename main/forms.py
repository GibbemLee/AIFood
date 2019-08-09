# -*- coding: utf-8 -*-
from django import forms

class UserForm(forms.Form):
    user_id = forms.CharField(label='Id', max_length=100)
    email = forms.EmailField(label='E-mail')
    pw = forms.CharField(label='Password', widget=forms.PasswordInput())
    pw2 = forms.CharField(label='Confirm_Password', widget=forms.PasswordInput())
    
class SignInForm(forms.Form):
    user_id = forms.CharField(label='Id', max_length=100)
    pw = forms.CharField(label='Password', widget=forms.PasswordInput())

from django.urls import path
from . import views

urlpatterns = [
    path('',views.login,name='login'),
    path('add',views.add,name='add'),
    path('image',views.fn_image,name='image'),
    path('video',views.video,name='video'),
    path('index_views',views.index_views,name='index_views'),
    path('signup',views.signup,name='signup'),

]  
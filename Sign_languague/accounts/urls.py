from django.urls import path
from accounts import views

urlpatterns =[
   path('signup/',views.RegisterAccount,name='signup'),
   path('login/',views.LoginAccount,name='login'),
   path('homepage/',views.HomePage,name='home'),
   path('logout/',views.logout_view,name='logout'),
   # path('streaming/',views.streaming_view,name='streaming'),
   path('video_feed/',views.video_feed,name='video_feed'),
   path('api/get_user_history/', views.get_user_history, name='get_user_history'),
   path('courses/',views.courses_view,name='courses'),
   path('get_courses/',views.get_courses,name = 'get_courses'),
]
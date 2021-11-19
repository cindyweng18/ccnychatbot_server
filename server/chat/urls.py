from django.urls import path
from . import views

urlpatterns = [
    path ('', views.index, name = 'index'),

    # User Message path
    path ('api/message-list/', views.MessageList.as_view()),
    path ('api/message-detail/', views.MessageDetail.as_view()),
    
    # Bot Message path
    path ('api/botmessage-list/', views.BotMessageList.as_view()),
]
from django.urls import re_path, path
from django.conf.urls import url
from . import consumers

# Call this the websocket_urlpatterns because it will be easier to understand whether it's a normal HTTP path or websocket path
websocket_urlpatterns = [
    # url (r'^ws/chat/(?P<room_name>[^/]+)', consumers.ChatRoomConsumer.as_asgi())
    # path ('ws/chat/<str:room_name>', consumers.ChatRoomConsumer.as_asgi())
    re_path (r"^ws/chat/(?P<room_name>[^/]+)/$", consumers.ChatRoomConsumer.as_asgi())
]
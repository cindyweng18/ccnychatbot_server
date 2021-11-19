"""
This file is very similar to the urls.py in the core django project
The only difference is this file will be called when the server is trying to make 
a Websocket request instead of a normal HTTP request
Hence, this file will point to which file to look for Websocket requests
but the logic for those Websocket routes will be defined in a consumers.py file
"""

from channels.auth import AuthMiddlewareStack
from channels.routing import ProtocolTypeRouter, URLRouter
import chat.routing

application = ProtocolTypeRouter ({
    # AuthmiddlewareStack is used in case we want to use authentication for the websocket
    'websocket': AuthMiddlewareStack (
        URLRouter (
            chat.routing.websocket_urlpatterns
        )
    )
})


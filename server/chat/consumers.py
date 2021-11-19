"""
A Channel is a mailbox where messages can be sent to
A Group is a group of related channels. Anyone who has the name of a group
can add/remove a channel to the group by name and send a message to all 
channels in the group

Every consumer instance of ChatRoomConsumer has an automatically generated
unique channel name and so can be communicated via a channel layer

In our application we want to have mutiple instances of ChatRoomConsumer
in the same room communicate with each other. To do that we will have
each ChatRoomConsumer add its channel to a group whose name is based
on the room name. That will allow ChatRoomConsumer to transmit messages
to all other ChatRoomConsumers in the same room
"""

from channels.generic.websocket import AsyncWebsocketConsumer
import json

class ChatRoomConsumer (AsyncWebsocketConsumer):

    # Create the Websocket connection
    async def connect(self):

        # get the room_name from the url path that opened the WebSocket connection to the ChatRoomConsumer
        self.room_name = self.scope ['url_route']['kwargs']['room_name']

        # Create a Channel Group based on the user-specified room_name
        self.room_group_name = 'chat_%s' % self.room_name

        # Add the room_group_name to the channel layer
        await self.channel_layer.group_add (
            self.room_group_name,
            self.channel_name
        )

        # Accept the Websocket connection
        await self.accept ()



    # Disconnect the Websocket connection
    async def disconnect(self, code):
        await self.channel_layer.group_discard (
            self.room_group_name,
            self.channel_name
        )


    #  Receive the messages 
    async def receive(self, text_data):
        # text_data has a key called message and name coming in from the frontend
        text_data_json = json.loads (text_data)
        message = text_data_json['message']
        name = text_data_json['name']

        # Broadcast the message to the group
        await self.channel_layer.group_send (
            # Room/Group we are broadcasting the message to
            self.room_group_name,
            # Payload
            {
                'type': 'chatroom_message',     # this has to match the function name
                'message': message ,         # name of the event is message
                'name': name
            }
        )
    # Get the message from the front end and do stuff with it
    async def chatroom_message (self, event):
        message = event ['message']
        name = event['name']

        await self.send (text_data=json.dumps ({
            'message': message,
            'name': name
        }))
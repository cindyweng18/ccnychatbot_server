from .models import Message, BotMessage
from rest_framework import serializers


# Create a User Message Serializer
class MessageSerializer (serializers.ModelSerializer):
    class Meta:
        model = Message
        fields = ['id', 'value']



# Create a Bot Message Serializer
class BotMessageSerializer (serializers.ModelSerializer):
    class Meta:
        model = BotMessage
        fields = ['id', 'value']
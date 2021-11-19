from django.shortcuts import render
from .models import Message, BotMessage
from .serializers import MessageSerializer, BotMessageSerializer
from django.http import Http404
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.views.decorators.csrf import csrf_exempt
import DistilBertModel.OurModel.testModel


# Create your views here.
def index (request):
    return render (request, 'index.html')


# Class Based Serializer Views
# User Message Views
class MessageList (APIView):
    """
    List all messages or create new messages 
    """
    def get (self, request, *args, **kwargs):
        """
        List all messages
        """
        messages = Message.objects.all ()
        serializer = MessageSerializer (messages, many = True)
        return Response (serializer.data)

    @csrf_exempt
    def post (self, request):
        """
        Create new messages
        """
        # Get the body of the data
        data = request.data    

        # Serialize the data
        serializer = MessageSerializer (data = data)
        if serializer.is_valid():
            serializer.save()
            return Response ({"Reponse": "New Message Created"}, status = status.HTTP_201_CREATED)
        return Response (serializer.errors, status = status.HTTP_400_BAD_REQUEST) 


class MessageDetail (APIView):
    """
    Show message details
    """
    def get_object (self, pk):
        try:
            return Message.objects.get (pk = pk)
        except Message.DoesNotExist:
            return Http404 

    def get (self):
        try:
            messages = Message.objects.filter (pk = pk)
            serializer = MessageSerializer (messages, many = True)
            return Response (serializer.data)
        except:
            return Response ({"Response": "Invalid Message ID/ Room ID"}, status = status.HTTP_400_BAD_REQUEST)



# Bot Message Views
class BotMessageList (APIView):
    """
    List all bot messages or create new bot messages 
    """
    def get (self, request, *args, **kwargs):
        """
        List all messages
        """
        bot_messages = BotMessage.objects.all ()
        serializer = BotMessageSerializer (bot_messages, many = True)
        return Response (serializer.data)

    @csrf_exempt
    def post (self, request):
        """
        Create new bot messages
        """
        # Get the body of the data
        data = request.data    

        #  Take the data from the body and call the ML model and get its response
        print (data)

        # Serialize the response
        serializer = BotMessageSerializer (data = data)
        if serializer.is_valid():
            serializer.save()
            return Response ({"Reponse": "New Bot Message Created"}, status = status.HTTP_201_CREATED)
        return Response (serializer.errors, status = status.HTTP_400_BAD_REQUEST)
from django.db import models
from django.utils import timezone


# Create your models here.

# User Message Schema
class Message (models.Model):
    value = models.CharField (max_length = 1000000)             # message can have a length of 1 million characters
    def __str__ (self):
        return f"Value: {self.value}"


# Bot Message Schema
class BotMessage (models.Model):
    value = models.CharField (max_length = 1000000)             # message can have a length of 1 million characters
    def __str__ (self):
        return f"Value: {self.value}"
    
    
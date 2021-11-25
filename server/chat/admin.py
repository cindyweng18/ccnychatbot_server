from django.contrib import admin
from .models import Message, BotMessage

# Register your models here.

admin.site.register (Message)
admin.site.register (BotMessage)


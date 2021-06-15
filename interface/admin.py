from django.contrib import admin
from .models import Request, Image, Caption, APIKey
# Register your models here.

admin.site.register(Request)
admin.site.register(Image)
admin.site.register(Caption)
admin.site.register(APIKey)
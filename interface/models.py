from django.db import models
from django.contrib.auth.models import User

class Request(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE,related_name="user")
    CHOICES=(
        (1,"Submitted"),
        (2,"Processing"),
        (3,"Completed"),
        (4,"Expired"))
    status = models.IntegerField(default=1, choices=CHOICES)
    filename = models.CharField(max_length=30)
    
class Image(models.Model):
    request = models.ForeignKey(Request, on_delete=models.CASCADE, related_name="image")
    location = models.CharField(max_length=50)

class Caption(models.Model):
    image = models.ForeignKey(Image, on_delete=models.CASCADE, related_name="caption")
    caption = models.CharField(max_length=100)
    probability = models.DecimalField(max_digits=10, decimal_places=9)

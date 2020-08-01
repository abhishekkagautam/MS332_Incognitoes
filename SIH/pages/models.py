from django.db import models

# Create your models here.
class labels(models.Model):
    time_stamp=models.CharField(max_length=100)
    gender = models.CharField(max_length=100) 
    color = models.CharField(max_length=120)
    pattern =models.CharField(max_length=100)
    cloths = models.TextField()
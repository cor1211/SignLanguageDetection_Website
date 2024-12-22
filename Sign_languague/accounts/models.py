from django.db import models
from django.contrib.auth.models import User
# Create your models here.

class History(models.Model):
   # Create ForeignKey
   user = models.ForeignKey(User,on_delete=models.CASCADE,related_name='histories')
   word = models.CharField(max_length=255)
   timestamp = models.DateTimeField(auto_now_add=True)
   
   def __str__(self):
      return f'{self.user.username} - {self.word} at {self.timestamp}'

class Courses(models.Model):
   image = models.ImageField(upload_to='courses/images/')
   title = models.CharField(max_length=255)
   rating = models.DecimalField(max_digits=2,decimal_places=1)
   reviews =models.PositiveBigIntegerField()
   start_date = models.DateField()
   time = models.CharField(max_length=255)
   sale_price = models.DecimalField(max_digits=8,decimal_places=2)
   original_price = models.DecimalField(max_digits=8,decimal_places=2)
   
   def __str__(self):
      return self.title
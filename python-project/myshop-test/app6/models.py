from django.db import models
from django.contrib.auth.models import AbstractBaseUser
class MyUser(AbstractBaseUser):
    identifier = models.CharField(max_length=40, unique=True)
    USERNAME_FIELD = 'identifier'
    photo=models.CharField('用户头像',max_length=50)
    weChat=models.CharField('微信',max_length=30)
    level=models.CharField('用户等级',max_length=1)
    def __str__(self):
        return self.username

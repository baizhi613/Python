from django.db import models

# Create your models here.

class Student(models.Model):
    name=models.CharField(max_length=100,verbose_name="姓名")
    sex=models.BooleanField(default=1,verbose_name="性别")
    age=models.IntegerField(verbose_name="年龄")
    class_null=models.CharField(max_length=5,verbose_name="班级编号")

    class Meta:
        db_table="tb_student"
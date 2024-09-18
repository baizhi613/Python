from django.db import models
class ImgFile(models.Model):
    name=models.CharField(verbose_name='名称',default='',max_length=30)
    headimg=models.FileField(verbose_name='文件名',upload_to="uploads/")
    def __str__(self):
        return str(self.name)
    class Meta:
        verbose_name='用户头像信息'
        db_table='user_img'

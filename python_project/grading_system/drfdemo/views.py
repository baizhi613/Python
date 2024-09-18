from django.shortcuts import render,HttpResponse

# Create your views here.
from rest_framework.views import APIView
from drfdemo.models import Student
from rest_framework import serializers
from rest_framework.response import Response

class StudentSerializer(serializers.Serializer):
    name=serializers.CharField()
    sex=serializers.BooleanField()
    age=serializers.IntegerField()
    class_null=serializers.CharField()

class StudentView(APIView):
    def get(self,request):
        students=Student.objects.all()
        serializer=StudentSerializer(instance=students,many=True)
        return Response(serializer.data)
    def post(self,request):
        print(request.data,type(request.data))
        serializer=StudentSerializer(data=request.data)
        # if serializer.is_valid():
        #     pass
        # else:
        #     return Response(serializer.errors)
        try:
            serializer.is_valid(raise_exception=True)
            print("validate_data",serializer.validated_data)
            stu=Student.objects.create(**serializer.validated_data)
            ser=StudentSerializer(instance=stu,many=False)
            return Response(ser.data)
        except:
            return Response(serializer.errors)
    
class StudentDetailView(APIView):
    def get(self,request,id):
        student=Student.objects.get(pk=id)
        serializer=StudentSerializer(instance=student,many=False)
        return Response(serializer.data)
    def delete(self,request,id):
        Student.objects.get(pk=id).delete()
        return Response()
    def put(self,request,id):
        serializer=StudentSerializer(data=request.data)
        try:
            serializer.is_valid(raise_exception=True)
            print("validate_data",serializer.validated_data)
            n=Student.objects.filter(pk=id).update(**serializer.validated_data)
            print(n)
            stu=Student.objects.get(id=id)
            ser=StudentSerializer(instance=stu,many=False)
            return Response(ser.data)
        except:
            return Response(serializer.errors)
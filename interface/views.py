from os import path
from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.views import View
from django.core.files.storage import default_storage
from django.http import JsonResponse, HttpResponseRedirect
from django.contrib.auth.models import User
from django.urls import reverse
from .models import Request, Image, Caption
from os.path import join
from threading import Thread
from django.contrib.auth.mixins import LoginRequiredMixin
from .model import model_thread_target
from image_cap.settings import BASE_DIR


@login_required
def index(request):
    return render(request, "interface/home.html")

class register(View):
    def get(self, request):
        return render(request, "registration/signup.html")
    def post(self, request):
        first_name=request.POST["fname"]
        last_name=request.POST["lname"]
        username=request.POST["username"]
        email=request.POST["email"]
        password=request.POST["password"]
        passwordconf=request.POST["passwordconf"]
        if password != passwordconf:
            messages.add_message(request, messages.ERROR, "Passwords do not match")
            return HttpResponseRedirect(reverse("register"))
        try:
            User.objects.create_user(username=username, email=email, password=password, first_name=first_name, last_name=last_name)
        except:
            messages.add_message(request, messages.ERROR, "This username already exists!")
            return HttpResponseRedirect(reverse("register"))
        messages.add_message(request, messages.SUCCESS, "User successfully registered! Login now...")
        return HttpResponseRedirect("/accounts/login/")

class predict(LoginRequiredMixin, View):
    def get(self, request):
        return render(request, "interface/predict.html")
    def post(self, request):
        print(request.FILES)
        req = Request(user=request.user, status=1, filename="temp")
        req.save()
        filenames = []
        for i, file in enumerate(request.FILES):
            file_name = default_storage.save(join(f'{req.pk}',f'{i+1}.jpg'), request.FILES[file])
            print(file_name)
            image = Image(request=req, location=file_name)
            image.save()
            filenames.append([image.pk, join(BASE_DIR,'media',file_name)])
        req.filename = req.pk
        req.save()
        t = Thread(target=model_thread_target, args=[req.pk, filenames], daemon=True)
        t.start()
        return JsonResponse({'success':True, "id": req.pk})

@login_required
def processing(request):
    requests = Request.objects.all().order_by('-pk')
    return render(request, "interface/processing.html", {"requests": requests})

@login_required
def result(request, ID):
    req = Request.objects.get(pk=ID)
    if req.status != 3:
        messages.add_message(request, messages.ERROR, "This request is not ready yet")
        return HttpResponseRedirect(reverse("processing"))
    images = Image.objects.all().filter(request=req)
    result = []
    for i in images:
        captions = Caption.objects.all().filter(image=i)
        caps = []
        for j in captions:
            caps.append([j.caption, j.probability])
        result.append(['/media/'+i.location, caps])
    return render(request, "interface/result.html", {"result": result})


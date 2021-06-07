from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.contrib import messages
from django.views import View
from django.core.files.storage import default_storage
from django.http import JsonResponse, HttpResponseRedirect
from django.contrib.auth.models import User
from django.urls import reverse
from secrets import token_urlsafe
from .models import Request, Image, Caption, APIKey
from os.path import join
from threading import Thread
from django.contrib.auth.mixins import LoginRequiredMixin
from .model import model_thread_target, apiModelRequest
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

@method_decorator(csrf_exempt, name='dispatch')
class API(View):
    def get(self, request):
        return JsonResponse({'error':'POST your image to this url with your API key'})
    def post(self, request):
        if "key" not in request.POST:
            return JsonResponse({'error':'Include a generated API key!'})
        try:
            APIKey.objects.get(key=request.POST["key"])
        except:
            return JsonResponse({'error':'This is not a valid API key!'})
        if "image" not in request.FILES:
            return JsonResponse({'error':'Include an image in POST'})
        print(request.FILES["image"])
        if ('jpeg' not in str(request.FILES["image"])) and ('jpg' not in str(request.FILES["image"])):
            return JsonResponse({'error':'Only JPG files allowed!'})
        file_name = default_storage.save('api.jpg', request.FILES["image"])
        imageLocation = join(BASE_DIR,'media',file_name)
        captions = apiModelRequest(imageLocation)
        return JsonResponse({'Captions': captions})

class apiEndpoint(LoginRequiredMixin, View):
    def get(self, request):
        try:
            key = APIKey.objects.get(username=request.user.username)
        except:
            key = False
        print (request.user.username)
        if key:
            return render(request, "interface/api-end.html", {"key":key.key})
        else:
            return render(request, "interface/api-end.html")
    def post(self, request):
        try:
            key = APIKey.objects.get(username=request.user.username)
        except:
            key = APIKey(username=request.user.username, key=token_urlsafe(20))
            key.save()
        print(key.key)
        return HttpResponseRedirect(reverse('api-end'))
        
@login_required
def tutorial(request):
    return render(request, "interface/tutorial.html")

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


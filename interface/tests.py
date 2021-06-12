from django.test import TestCase, Client
from .models import Image, Caption, APIKey, Request
from django.contrib.auth.models import User
from os.path import join
from image_cap.settings import BASE_DIR
from .model import apiModelRequest

class CheckIfDatabaseModelsWork(TestCase):
    def test_models(self):
        try:
            user = User.objects.create_user('testuser', 'testuser@email.com', 'password')
            request = Request(user=user, status=1, filename="temp")
            request.save()
            test_image = join(BASE_DIR, 'interface', 'test_images', 'bike.jpeg')
            image = Image(request=request, location=test_image)
            image.save()
            caption = Caption(image=image, caption="Test caption", probability=0.2650)
            caption.save()
            apikey = APIKey(username=user.username, key="testKey908309849389394893klfjlsk")
            apikey.save()
            print("Models were initialized correctly")
        except:
            raise AssertionError("The models were not initialized correctly")

class CheckIfModelIsPredicting(TestCase):
    def test_predict(self):
        try:
            test_image = join(BASE_DIR, 'interface', 'test_images', 'bike.jpeg')
            captions = apiModelRequest(test_image)
            print("Captions generated were: ", captions)
        except:
            raise AssertionError("Captions could not be generated correctly")

class CheckIfWebPagesLoadCorrectly(TestCase):
    def test_webpages(self):
        user = User.objects.create_user('testuser', 'testuser@email.com', 'password')
        c = Client()
        c.login(username='testuser', password='password')
        response = c.get('/')
        self.assertEqual(response.status_code, 200)
        response = c.get('/predict/')
        self.assertEqual(response.status_code, 200)
        response = c.get('/processing/')
        self.assertEqual(response.status_code, 200)
        response = c.get('/api-end/')
        self.assertEqual(response.status_code, 200)
        response = c.get('/tutorial/')
        self.assertEqual(response.status_code, 200)
        print("Webpages loaded successfully")
from django.apps import AppConfig
#from .model import Model

class InterfaceConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'interface'

# class ModelObject(AppConfig):
#     def ready(self):
#         model = Model()
#         model.build()
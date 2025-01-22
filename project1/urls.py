from django.contrib import admin
from django.urls import include, path
from uiApp import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('uiApp.urls')),
]

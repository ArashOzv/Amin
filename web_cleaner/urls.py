from django.contrib import admin
from django.urls import path

from web_cleaner.core import views

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", views.index, name="index"),
    path("api/clean/", views.clean_selection, name="clean_selection"),
]

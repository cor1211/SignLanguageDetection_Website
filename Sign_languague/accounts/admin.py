from django.contrib import admin
from .models import Courses
from .models import History
# Register your models here.
admin.site.register(History)
admin.site.register(Courses)
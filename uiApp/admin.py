from django.contrib import admin
from .models import SpikeData

class SpikeDataAdmin(admin.ModelAdmin):
    list_display = ('dateTime', 'value')  # Columns to display in the list view
    search_fields = ('dateTime',)  # Enable search functionality for the dateTime field
    list_filter = ('value',)  # Add filtering options for the value field
    ordering = ('dateTime',)  # Default ordering by dateTime

# Register the SpikeData model with the admin site
admin.site.register(SpikeData, SpikeDataAdmin)

from django.contrib import admin
from .models import StationName

@admin.register(StationName)
class StationNameAdmin(admin.ModelAdmin):
    list_display = ('station_name',)

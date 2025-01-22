from django.db import models


from django.db import models

class SpikeData(models.Model):
    dateTime = models.CharField(max_length=100)  # To store dateTime as string
    value = models.FloatField(null=True, blank=True)  # To store the original numeric value
    modified_value = models.FloatField(null=True, blank=True)  # To store modified values after processing

    def __str__(self):
        return f"{self.dateTime}: {self.value} (Modified: {self.modified_value})"


from django.db import models

class StationName(models.Model):
    station_name = models.CharField(max_length=255)

    def __str__(self):
        return self.station_name
    

from django.db import models

from django.db import models

class StationRecord(models.Model):
    station_id = models.CharField(max_length=50)  # Station identifier
    recorded_highest_wl = models.FloatField()  # Highest water level
    recorded_lowest_wl = models.FloatField()  # Lowest water level

    def __str__(self):
        return self.station_id
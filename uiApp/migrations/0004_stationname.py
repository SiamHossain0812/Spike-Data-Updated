# Generated by Django 5.1 on 2024-12-29 11:03

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('uiApp', '0003_delete_datafilling_delete_raindata_and_more'),
    ]

    operations = [
        migrations.CreateModel(
            name='StationName',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('station_name', models.CharField(max_length=255)),
            ],
        ),
    ]

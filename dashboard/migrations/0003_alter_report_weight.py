# Generated by Django 4.0.5 on 2022-06-18 04:38

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('dashboard', '0002_report_image'),
    ]

    operations = [
        migrations.AlterField(
            model_name='report',
            name='weight',
            field=models.DecimalField(blank=True, decimal_places=2, default=0, max_digits=10, null=True),
        ),
    ]
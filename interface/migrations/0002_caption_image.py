# Generated by Django 3.2.3 on 2021-05-28 09:20

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('interface', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Image',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('location', models.CharField(max_length=50)),
                ('request', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='image', to='interface.request')),
            ],
        ),
        migrations.CreateModel(
            name='Caption',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('caption', models.CharField(max_length=100)),
                ('probability', models.DecimalField(decimal_places=9, max_digits=10)),
                ('image', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='caption', to='interface.image')),
            ],
        ),
    ]

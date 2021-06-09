# Generated by Django 3.2.4 on 2021-06-07 08:06

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('auth', '0012_alter_user_first_name_max_length'),
        ('interface', '0002_caption_image'),
    ]

    operations = [
        migrations.CreateModel(
            name='APIKey',
            fields=[
                ('user', models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, primary_key=True, serialize=False, to='auth.user')),
                ('key', models.CharField(max_length=50)),
            ],
        ),
    ]
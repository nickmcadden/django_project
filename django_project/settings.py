"""
Django settings for django_project project.

Generated by 'django-admin startproject' using Django 4.2.4.

For more information on this file, see
https://docs.djangoproject.com/en/4.2/topics/settings/

For the full list of settings and their values, see
https://docs.djangoproject.com/en/4.2/ref/settings/
"""

from pathlib import Path
import os

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = os.path.join(BASE_DIR, 'data')

# The forecasts will be more accurate if the locations for the weather data are optimised for the type of renewable
# More locations will need more API calls so try and keep to 10 locations for each type
LOCATION_GEOCODE_DATA = {}

LOCATION_GEOCODE_DATA['v1_locations'] = {'Bracknell' : (51.4136, -0.7505), 
                         'Cardiff': (51.48, -3.18),
                         'Leeds' : (53.7965, -1.5478),
                         'Belfast': (54.5968, -5.9254),
                         'Edinburgh': (55.9521, -3.1965),
                         'Inverness': (57.4791, -4.224),
                         'Norwich': (52.6278, 1.2983),
                         'Hull': (53.7446, -0.3352),
                         'Carlisle': (54.8951, -2.9382)}

l_offset = 1
for i in LOCATION_GEOCODE_DATA['v1_locations'].keys():
    LOCATION_GEOCODE_DATA['v1_locations'][i] = (LOCATION_GEOCODE_DATA['v1_locations'][i][0]+l_offset, LOCATION_GEOCODE_DATA['v1_locations'][i][1]+l_offset)

# Offshore is mainly produced off the east coast of the country                      
LOCATION_GEOCODE_DATA['wind(offshore)'] = {'Moray East' : (58.1, -2.8), 
                         'Walney': (54.05, -3.516),
                         'Gwynty Mor' : (53.45, -3.583),
                         'Rampion': (50.6, -0.266),
                         'Lynn': (53.12, 0.436),
                         'Triton Knoll': (53.5, 0.8),
                         'Hornsea': (53.8, 1.791),
                         'Thanet': (51.4, 1.633),
                         'Kentish Flats': (51.46, 1.09)}

# Onshore wind production is most concentrated in Scotland and the North of England    
LOCATION_GEOCODE_DATA['wind(onshore)'] = {'Plymouth' : (50.37, -4.143), 
                         'Cardiff': (51.48, -3.18),
                         'Cambridge': (52.20, 0.119),
                         'Hull': (53.74, -0.3352),
                         'Leeds' : (53.79, -1.5478),
                         'Belfast': (54.59, -5.9254),
                         'Edinburgh': (55.95, -3.1965),
                         'Inverness': (57.47, -4.224),
                         'Kirkmuirhill': (55.65, -3.918),
                         'Carlisle': (54.8951, -2.9382)}

# Solar power is produced mostly in the south of England and the midlands
LOCATION_GEOCODE_DATA['solar'] = {'Plymouth' : (50.37, -4.143), 
                         'Cardiff': (51.48, -3.18),
                         'Bournemouth' : (50.71, -1.883),
                         'Dover': (51.12, 1.316),
                         'Oxford': (51.75, -1.257),
                         'Cambridge': (52.20, 0.119),
                         'Mansfield': (53.13, -1.200),
                         'Liverpool': (53.40, -2.983),
                         'Belfast': (54.8951, -2.9382),
                         'Edinburgh': (55.9521, -3.1965)}

HOURLY_WEATHER_VARIABLES = ['windspeed_10m','winddirection_10m','cloudcover','surface_pressure','temperature_2m','precipitation','rain','terrestrial_radiation']

# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/4.2/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = 'django-insecure-j4@q45y7(bv*sk3b$qa7076-x-l70k9wyv_#zr0gfk!uw#4n&b'

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

ALLOWED_HOSTS = ['ec2-54-147-197-190.compute-1.amazonaws.com', '127.0.0.1']
# Application definition

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'main',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'django_project.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'django_project.wsgi.application'

# Database
# https://docs.djangoproject.com/en/4.2/ref/settings/#databases

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}


# Password validation
# https://docs.djangoproject.com/en/4.2/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]


# Internationalization
# https://docs.djangoproject.com/en/4.2/topics/i18n/

LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'UTC'

USE_I18N = True

USE_TZ = True


# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/4.2/howto/static-files/

STATIC_URL = 'static/'
STATICFILES_DIRS = [os.path.join(BASE_DIR, "django_project", "static"),]

# Default primary key field type
# https://docs.djangoproject.com/en/4.2/ref/settings/#default-auto-field

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

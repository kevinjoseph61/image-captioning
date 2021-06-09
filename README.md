# Image-Caption-Generator

Predict image captions using an interactive Django interface

## `master branch`
Master branch is for developement Django server to be run on local system (Windows). Follow these steps to use:  
1. Install all the dependencies by running `pipenv install` (install pipenv first)  
2. Download the model files using the links hosted by my Google Drive in setup.sh in `prod` branch

## `prod branch`
[![Azure - image-caption-generator, Build and Deploy](https://github.com/kevinjoseph61/image-captioning/actions/workflows/prod_image-caption-generator.yml/badge.svg)](https://github.com/kevinjoseph61/image-captioning/actions/workflows/prod_image-caption-generator.yml)  
Prod branch was made keeping in mind the end result of deployment on Azure using the B3 (7GB) App Service plan on Linux with Python 3.7. You can view the version that I have deployed [here](https://image-caption-generator.azurewebsites.net/)
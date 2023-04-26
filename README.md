# Detect Stopped vehicles In Video

Hi,

In this repo, You will see how to detect vehicles that stopped in a given video.

## Step 1:
Install all requirements
`pip install -r requirements.txt`

## Step 2: Run the `python` file
`python stopped_car_detect.py`

## Added Feature
You can now run the docker file directly and it will give you the exact same result

`docker build -t stopped_car .`
then after building
`docker run stopped_car` 

### Side Notes:
1. I want to add the functionality to save images of vehicles that have stopped into a folder.
2. I want to dockerize this file - Done
3. Change the image on the readme file
4. Remove TG from sort and remove skimage from requirements file - Done


![Header](print.png)

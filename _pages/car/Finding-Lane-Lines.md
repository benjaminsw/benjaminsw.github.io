---
permalink: /car/Finding-Lane-Lines/
header:
  image: "/images/digital-transition2.jpg"
---
<h2 id="top"></h2>

# Self-Driving Car Engineer


### **Finding Lane Lines on the Road**
***
In this project, I will use the tools you learned about in the lesson to identify lane lines on the road.  I will develop the pipeline on a series of individual images, and later apply the result to a video stream (really just a series of images).

### Pipeline will be constructed as follow:
    1 Read in Colour Images
    2 Convert Them to Graysclae
    3 Apply Gaussian Blur to Remove Noise
    4 Use Canny to Detact Edges
    5 Find Region of Interest in The Images
    6 Apply Hough Transform Line Detection
    7 Consolidate and Interpolatethe Hough Lines and Draw Them on Original Images.
    8 Try Apply The Pipeline to Process Video Clips to Find Lane Lines


Let's have a look at our first image called 'test_images/solidWhiteRight.jpg'. The 2 cells below will display the image.



<figure>
 <img src="examples/line-segments-example.jpg" width="380" alt="Combined Image" />
 <figcaption>
 <p></p>
 <p style="text-align: center;"> Our output should look something like this (above) after detecting line segments using the helper functions below </p>
 </figcaption>
</figure>
 <p></p>
<figure>
 <img src="examples/laneLines_thirdPass.jpg" width="380" alt="Combined Image" />
 <figcaption>
 <p></p>
 <p style="text-align: center;"> Our goal is to connect/average/extrapolate line segments to get output like this</p>
 </figcaption>
</figure>

## Import Packages


```python
#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
%matplotlib inline
```

## Read in an Image


```python
#reading in an image
image = mpimg.imread('test_images/solidWhiteRight.jpg')

#printing out some stats and plotting
print('This image is:', type(image), 'with dimensions:', image.shape)
plt.imshow(image)  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')
```

    This image is: <class 'numpy.ndarray'> with dimensions: (540, 960, 3)





    <matplotlib.image.AxesImage at 0x7ff59bec8198>




![png](/images/Finding-Lane-Lines/output_5_2.png)


## Ideas for Lane Detection Pipeline

**Some OpenCV functions (beyond those introduced in the lesson) that might be useful for this project are:**

`cv2.inRange()` for color selection  
`cv2.fillPoly()` for regions selection  
`cv2.line()` to draw lines on an image given endpoints  
`cv2.addWeighted()` to coadd / overlay two images
`cv2.cvtColor()` to grayscale or change color
`cv2.imwrite()` to output images to file  
`cv2.bitwise_and()` to apply a mask to an image

**Check out the OpenCV documentation to learn about these and discover even more awesome functionality!**

## Helper Functions

Below are some helper functions to help get us started.


```python
import math

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming our grayscaled image is called 'gray')
    Then I will call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if an image is read with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)   

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
    """
    NOTE: this is the function we want to use as a starting point once we want to
    average/extrapolate the line segments we will detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If we want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below


    """

    right_lane = []
    left_lane = []

    height, width, channels = img.shape
    half_height = height # in other words, it is for keeping the position of y in the image

    for line in lines:
        for x1,y1,x2,y2 in line:

            # compute slope of the lines
            slope = (y2-y1)/(x2-x1)

            if 0 > slope > np.NINF: # left line
                left_lane.append((slope, x1, y1, y1 - slope*x1)) # for the first x,y pair, keep slope, x, y, intercept
                left_lane.append((slope, x2, y2, y2 - slope*x2)) # for the second x,y pair, keep slope, x, y intercept
            elif 0 < slope < np.PINF: # right line
                right_lane.append((slope, x1, y1, y1 - slope*x1)) # for the first x,y pair, keep slope, x, y, interecpt
                right_lane.append((slope, x2, y2, y2 - slope*x2)) # for the second x,y pair, keep slope, x, y, intercept

            # find furthest line of the lane in the mid image
            half_height = min(y1,y2,half_height)

    # get to draw the left line
    left_slope = list(np.mean(left_lane, axis=0))[0] # get left slope
    left_intercept = list(np.mean(left_lane, axis=0))[3] # get left intercept
    x_min_left = int((half_height-left_intercept)/left_slope) # get top x left
    x_max_left = int((height-left_intercept)/left_slope) # get bottom x left
    cv2.line(img, (x_min_left, half_height), (x_max_left, height), color, thickness) # create a left line

    # get to draw the right line
    right_slope = list(np.mean(right_lane, axis=0))[0] # get right slope
    right_intercept = list(np.mean(right_lane, axis=0))[3] # get right intercept
    x_min_right = int((half_height-right_intercept)/right_slope) # get top x right
    x_max_right = int((height-right_intercept)/right_slope) # get bottom x right
    cv2.line(img, (x_min_right, half_height), (x_max_right, height), color, thickness) # create a right line


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines) # call the function above to draw lines
    return line_img

# Python 3 has support for cool math symbols.
def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)
```

## Test Images

Build our pipeline to work on the images in the directory "test_images"  
**We should make sure our pipeline works well on these images before we try the videos.**


```python
import os
os.listdir("test_images/")
```




    ['solidYellowLeft.jpg',
     'whiteCarLaneSwitch.jpg',
     'solidWhiteRight.jpg',
     'solidYellowCurve.jpg',
     'solidYellowCurve2.jpg',
     'solidWhiteCurve.jpg']



## Build a Lane Finding Pipeline



Build the pipeline and run our solution on all test_images. Make copies into the `test_images_output` directory.

Try tuning the various parameters, especially the low and high Canny thresholds as well as the Hough lines parameters.


```python
# read file name from "test_images" directory
files = os.listdir("test_images/")
# set figure size as well as representation 3 by 2
fig = plt.figure(figsize=(8, 8))
columns = 2
rows = 3

for i, file in enumerate(files):
    # read image in
    img = mpimg.imread("test_images/"+file)
    gray = grayscale(img)
    gray = gaussian_blur(gray, 3)
    edges = canny(gray, 50, 150)

    imshape = img.shape
    vertices = np.array([[(0.51*imshape[1], imshape[0]*0.58), (0.49*imshape[1], imshape[0]*0.58), (0, imshape[0]), (imshape[1], imshape[0])]], dtype=np.int32)
    target = region_of_interest(edges, vertices)

    lines = hough_lines(target, 1, np.pi/180, 20, 20, 300)
    result = weighted_img(lines, img, 0.8, 1.0)

    # add index of the image to the grid to show images
    fig.add_subplot(rows, columns, i+1)
    plt.imshow(result, cmap='gray')

    r, g, b = cv2.split(result)
    result = cv2.merge((b,g,r))

    # write result
    cv2.imwrite("test_images_output/output_"+file, result)
# print images
plt.show()


```


![png](/images/Finding-Lane-Lines/output_15_0.png)


## Test on Videos

Now let's drawing lanes over video.

We can test our solution on two provided videos:

`solidWhiteRight.mp4`

`solidYellowLeft.mp4`


```python
import sys
!{sys.executable} -m pip install imageio-ffmpeg
```

    Collecting imageio-ffmpeg
      Downloading https://files.pythonhosted.org/packages/44/51/8a16c76b2a19ac2af82001985c80d3caca4c373528855cb27e12b39373fb/imageio-ffmpeg-0.3.0.tar.gz
    Building wheels for collected packages: imageio-ffmpeg
      Running setup.py bdist_wheel for imageio-ffmpeg ... [?25ldone
    [?25h  Stored in directory: /root/.cache/pip/wheels/54/ed/2d/4281f5e6a575bfaa7d8f1f4173a7cb635adc406c1f8d87bfc8
    Successfully built imageio-ffmpeg
    Installing collected packages: imageio-ffmpeg
    Successfully installed imageio-ffmpeg-0.3.0

import imageio
imageio.plugins.ffmpeg.download()

```python
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML
```


```python
def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)
    gray = grayscale(image)
    gray = gaussian_blur(gray, 3)
    edges = canny(gray, 50, 150)

    imshape = image.shape
    vertices = np.array([[(0.51*imshape[1], 0.58*imshape[0]), (0.49*imshape[1], 0.58*imshape[0]), (0, imshape[0]), (imshape[1], imshape[0])]], dtype=np.int32)
    target = region_of_interest(edges, vertices)

    lines = hough_lines(target, 1, np.pi/180, 20, 5, 2)
    result = weighted_img(lines, image, 0.8, 1.0)

    return result
```

Let's try the one with the solid white lane on the right first ...


```python
white_output = 'test_videos_output/solidWhiteRight.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
%time white_clip.write_videofile(white_output, audio=False)
```

    [MoviePy] >>>> Building video test_videos_output/solidWhiteRight.mp4
    [MoviePy] Writing video test_videos_output/solidWhiteRight.mp4


    100%|█████████▉| 221/222 [00:14<00:00, 15.59it/s]


    [MoviePy] Done.
    [MoviePy] >>>> Video ready: test_videos_output/solidWhiteRight.mp4

    CPU times: user 3.17 s, sys: 319 ms, total: 3.49 s
    Wall time: 16 s


Play the video inline, or if you prefer find the video in your filesystem (should be in the same directory) and play it in your video player of choice.


```python
HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(white_output))
```





<video width="960" height="540" controls>
  <source src="test_videos_output/solidWhiteRight.mp4">
</video>


*last edited: 10/06/19*

<a href="#top">Go to top</a>

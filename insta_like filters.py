#!/usr/bin/env python
# coding: utf-8

# # Image Processing 1 -Instagram Filters

#     علي ناصر الدين علي السيد الحملاوي 1400842

# ## 1- Documentation

#     I have made a three basic funs that control hue_sat, channel enhancement and brightness
#     then i have used these three funs to apply filter
#     without any idea about what the filter is, this have been made by a magic :D, no no no i am just kidding,
#     this realy have been made by trial and error algorithm 
#     Now, grab yourself a cup of coffee/tea/beer:D and enjoy 
#     btw, thanks for your time to read this

# ## 2-Script

# In[2]:


import cv2
get_ipython().magic(u'matplotlib inline')
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

def channel_enhance(img, channel, level=1.0):
    if channel == 'B':
        blue_channel = img[:, :, 0]
        blue_channel = blue_channel * level
        blue_channel = np.clip(blue_channel, 0, 255)
        img[:, :, 0] = blue_channel
    elif channel == 'G':
        green_channel = img[:, :, 1]
        green_channel = green_channel * level
        green_channel = np.clip(green_channel, 0, 255)
        img[:, :, 0] = green_channel
    elif channel == 'R':
        red_channel = img[:, :, 2]
        red_channel = red_channel * level
        red_channel = np.clip(red_channel, 0, 255)
        img[:, :, 0] = red_channel
    img = img.astype(np.uint8)
    return img


def hue_saturation(img_rgb, alpha=1, beta=1.0):
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2HSV)
    hue = img_hsv[:, :, 0]
    saturation = img_hsv[:, :, 1]
    hue = np.clip(hue * alpha, 0,  179)
    saturation = np.clip(saturation * beta, 0, 255)
    img_hsv[:, :, 0] = hue
    img_hsv[:, :, 1] = saturation
    img_transformed = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    return img_transformed


def brightness_contrast(img, alpha=1.0, beta=0):
    img_contrast = img * alpha
    img_bright = img_contrast + beta
    img_bright = np.clip(img_bright, 0, 255)
    img_bright = img_bright.astype(np.uint8)
    return img_bright


def insta_like(image, insta_filter):
    
      if insta_filter == "walden" :
        img = hue_saturation(source_img, 1, .1)
        img = brightness_contrast(source_img, 1.5, -30)
        return img
        
      elif insta_filter == "lily" :
            img = channel_enhance(image,"R", 1.2)
            img = channel_enhance(image,"G", .8)
            return img
      elif insta_filter == "lomo":
            image2 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img = hue_saturation(image2, 1, .1)
            img = brightness_contrast(image2, 1.5, -30)
            return img
      elif insta_filter == "prpocket":
            source_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img = channel_enhance(source_img,"R", 1.1)
            img = channel_enhance(source_img,"G", 1.8)
            return img


      elif insta_filter == "lordkelvin":
            img = channel_enhance(image,"R", 1.1)
            img = channel_enhance(image,"G", 1.3)
            return img


      elif insta_filter == "inkwell":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = channel_enhance(image,"R", 1.1)
            image = channel_enhance(image,"G", 1.3)
            image = hue_saturation(image, 1, .1)
            return image


# ### 2.0- Input Image

# In[11]:


original_image = cv2.imread("AI.jpg", 1)
original_image1 = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
plt.imshow(original_image1)
plt.show()


# ### 2.1-lily Fliter

# In[12]:


plt.imshow(insta_like(original_image,"lily"))
plt.show()


# ### 2.2-lomo Filter

# In[13]:


plt.imshow(insta_like(original_image,"lomo"))
plt.show()


# ### 2.3-prpocket Filter

# In[15]:


plt.imshow(insta_like(original_image,"prpocket"))
plt.show()


# ### 2.4-lordkelvin Filter

# In[17]:


plt.imshow(insta_like(original_image,"lordkelvin"))
plt.show()


# ### 2.5-inkwell Filter

# In[18]:


plt.imshow(insta_like(original_image,"inkwell"))
plt.show()


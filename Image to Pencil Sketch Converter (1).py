#!/usr/bin/env python
# coding: utf-8

# # KRUTIKA .D. NAIDU

# # * IMAGE TO PENCIL SKETCH USING PYTHON

# PROBLEM STATEMENT: We need to read the image in RBG format and then convert it to a grayscale image. This will turn an image into a classic black-and-white photo. Then the next thing to do is invert the grayscale image also called the negative image, this will be our inverted grayscale image. Inversion can be used to enhance details. Then we can finally create the pencil sketch by mixing the grayscale image with the inverted blurry image. This can be done by dividing the grayscale image by the inverted blurry image. Since images are just arrays, we can easily do this programmatically using the divide function from the cv2 library in Python. 

# In[9]:


import cv2


# In[10]:


image = cv2.imread(r"C:/Users/Krutika/Downloads/IMG-20230124-WA0024.jpg")
cv2.imshow("pa", image)
cv2.waitKey(0)


# In[11]:


gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("New Image", gray_image)
cv2.waitKey(0)


# In[12]:


inverted_image = 255 - gray_image
cv2.imshow("Inverted", inverted_image)
cv2.waitKey()


# In[13]:


blurred = cv2.GaussianBlur(inverted_image, (21, 21), 0)


# In[14]:


inverted_blurred = 255 - blurred
pencil_sketch = cv2.divide(gray_image, inverted_blurred, scale=256.0)
cv2.imshow("Sketch", pencil_sketch)
cv2.waitKey(0)


# In[15]:


cv2.imshow("original image",image)
cv2.imshow("pencil sketch",pencil_sketch)
cv2.waitKey(0)


# In[ ]:





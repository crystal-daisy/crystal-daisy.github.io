## Image Processing in Opencv



##### 1. 颜色空间转换

##### cv.cvtColor(input_image, flag)，其中flag确定转换类型

> ###### 对于BGR →灰色转换，我们使用标志cv.COLOR_BGR2GRAY
>
> ###### 对于BGR同样→HSV，我们使用标志cv.COLOR_BGR2HSV

```python
# 要获取其他标志，使用如下代码
flags = [i for i in dir(cv) if i.startswith('COLOR_')];
print(flags);
# ['COLOR_BAYER_BG2BGR', 'COLOR_BAYER_BG2BGRA', 'COLOR_BAYER_BG2BGR_EA', ...
```

```python
cap = cv.VideoCapture(0)
while(1):
    #取每一帧
    _, frame = cap.read() 
    # 调用cv.cvtColor将帧的颜色从BGR转化成HSV
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    # 定义HSV中蓝色的上下限值,用于取色
    lower_blue = np.array([50,255,255])
    upper_blue = np.array([70,255,255])
    # 从HSV图像中获取在蓝色范围内的图像显示
    mask = cv.inRange(hsv, lower_blue, upper_blue)
    # 与原始图像按位与 0xff
    res = cv.bitwise_and(frame,frame, mask= mask)
    cv.imshow('frame',frame)
    cv.imshow('mask',mask)
    cv.imshow('res',res)
    k = cv.waitKey(5) & 0xFF
    if k == 27:
        break
cv.destroyAllWindows()
```

![frame.jpg](https://docs.opencv.org/3.4.10/frame.jpg)

```python
# 要查找Green的HSV值
Green = np.uint8([[[0,255,0]]]);
hsv_green = cv.cvtColor(Green,cv.COLOR_BGR2HSV);
print(hsv_green);
# [[[60 255 255]]]
```

---



##### 2. 图像的几何变换

> ###### 平移、旋转、仿射变换等。
>
> 

###### 2.1 缩放 cv.resize

```python
# cv.resize(原图像，目标图像，目标图像尺寸，fx，fy，interpolation)
# interpolation取值为cv.INTER_AREA(仅用于缩小)，cv.INTER_CUBIC(缩放，速度慢)
# cv.INTER_LINEAR(缩放)
img = cv.imread('messi5.jpg')
res = cv.resize(img,None,fx=2, fy=2, interpolation = cv.INTER_CUBIC)
# 不指定输出目标图像尺寸，指定fx和fy，沿各方向将图像扩大两倍
#OR
height, width = img.shape[:2]
res = cv.resize(img,(2*width, 2*height), interpolation = cv.INTER_CUBIC)
# 不指定fx和fy，指定缩放到的具体尺寸
```

###### 2.2 Translation

$$
\mathbf{M}= \begin{vmatrix}
1 & 0 & t_x \\
0 & 1 & t_y \\
\end{vmatrix}
$$

$$
\mathbf{dst}=M*\begin{vmatrix}
x \\ y \\ 1 \\
\end{vmatrix}
$$

```python
# cv.warpAffine(原图像，变换矩阵M，目标图像尺寸)
img = cv.imread('messi5.jpg',0)
rows,cols = img.shape
M = np.float32([[1,0,100],[0,1,50]])
dst = cv.warpAffine(img,M,(cols,rows))
cv.imshow('img',dst)
cv.waitKey(0)
cv.destroyAllWindows()
```

###### 2.3 Rotation 

$$
\mathbf{M}= \begin{vmatrix}
cosθ & -sinθ \\
sinθ & cosθ \\
\end{vmatrix}
$$

```python
# 上面M是按照原点为中心进行旋转的，使用cv.getRotationMatrix2D可在任意位置旋转
img = cv.imread('messi5.jpg',0)
rows,cols = img.shape
# cv.getRotationMatrix2D(旋转中心，旋转角度，旋转后图像的缩放比例)
M = cv.getRotationMatrix2D(((cols-1)/2.0,(rows-1)/2.0),90,1)
dst = cv.warpAffine(img,M,(cols,rows))
```

###### 2.4 Affine Transformation 

```python
# 仿射变换，原始图像中的所有平行线在输出图像中仍将平行
img = cv.imread('drawing.png')
rows,cols,ch = img.shape
# M使用三个点求出
pts1 = np.float32([[50,50],[200,50],[50,200]])
pts2 = np.float32([[10,100],[200,50],[100,250]])
# cv.getAffineTransform(原始图像三个点的坐标,变换后的这三个点对应的坐标)自动生成仿射变换的M
M = cv.getAffineTransform(pts1,pts2)
dst = cv.warpAffine(img,M,(cols,rows))
plt.subplot(121),plt.imshow(img),plt.title('Input')
plt.subplot(122),plt.imshow(dst),plt.title('Output')
plt.show()
```

###### 2.5 Perspective Transformation

```python
# 在转换后，直线也将保持直线,需要四个点(其中三点不共线)
img = cv.imread('sudoku.png')
rows,cols,ch = img.shape
pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])
# M=3*3矩阵
M = cv.getPerspectiveTransform(pts1,pts2)
dst = cv.warpPerspective(img,M,(300,300))
plt.subplot(121),plt.imshow(img),plt.title('Input')
plt.subplot(122),plt.imshow(dst),plt.title('Output')
plt.show()
```

![image-20210104145400942](C:\Users\元宵\AppData\Roaming\Typora\typora-user-images\image-20210104145400942.png)

---



##### 3. 图像阈值

###### 3.1 简单阈值

```python
# cv.threshold(原图像，阈值，分配给超过阈值的像素的最大值，阈值类型)
# 对每个像素，若像素值小于阈值，则设置为XXX，否则设置为XXX
img = cv.imread('gradient.png',0)
ret,thresh1 = cv.threshold(img,127,255,cv.THRESH_BINARY)
ret,thresh2 = cv.threshold(img,127,255,cv.THRESH_BINARY_INV)
ret,thresh3 = cv.threshold(img,127,255,cv.THRESH_TRUNC)
ret,thresh4 = cv.threshold(img,127,255,cv.THRESH_TOZERO)
ret,thresh5 = cv.threshold(img,127,255,cv.THRESH_TOZERO_INV)
titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]
for i in xrange(6):
    plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()
```

![image-20210104153241923](C:\Users\元宵\AppData\Roaming\Typora\typora-user-images\image-20210104153241923.png)

效果图

![image-20210104153313939](C:\Users\元宵\AppData\Roaming\Typora\typora-user-images\image-20210104153313939.png)

###### 3.2 自适应阈值

```python
# cv.adaptiveThreshold(原图像，输出图像，向上最大值，
#		自适应方法[平均cv.ADAPTIVE_THRESH_MEAN_C/高斯
#		cv.ADAPTIVE_THRESH_GAUSSIAN_C]，阈值类型，块大小b，常量C)
# 自适应阈值通过计算b*b大小的像素块的加权均值-常量C得到，平均[则所有像素周围权值相同]，高斯[则周围像素权值根据其到中心点的距离通过高斯方程得到]
# 注意块大小b只能为奇数
img = cv.imread('sudoku.png',0)
# cv.medianBlur()中值滤波函数，中值滤波将图像的每个像素用邻域 (以当前像素为中心的正方形区域)像素的 中值代替
img = cv.medianBlur(img,5)
ret,th1 = cv.threshold(img,127,255,cv.THRESH_BINARY)
th2 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C,\
            cv.THRESH_BINARY,11,2)
th3 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv.THRESH_BINARY,11,2)
titles = ['Original Image', 'Global Thresholding (v = 127)',
            'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
images = [img, th1, th2, th3]
for i in xrange(4):
    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()
```

![image-20210104154940864](C:\Users\元宵\AppData\Roaming\Typora\typora-user-images\image-20210104154940864.png)

###### 3.3  Otsu's Binarization 大津二值化

```python
# 大津二值化算法，通过统计整个图像的直方图特性来实现全局阈值T的自动选取。
img = cv.imread('noisy2.png',0)
# 全局简单阈值
ret1,th1 = cv.threshold(img,127,255,cv.THRESH_BINARY)
# Otsu's 阈值
ret2,th2 = cv.threshold(img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
# Otsu's 阈值 + Gaussian滤波
# 高斯滤波：用一个模板（或称卷积、掩模）扫描图像中的每一个像素，用模板确定的邻域内像素的加权平均灰度值去替代模板中心像素点的值。
blur = cv.GaussianBlur(img,(5,5),0)
ret3,th3 = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
# 画出原噪声图，直方图，及处理后的图像
images = [img, 0, th1,
          img, 0, th2,
          blur, 0, th3]
titles = ['Original Noisy Image','Histogram','Global Thresholding (v=127)',
          'Original Noisy Image','Histogram',"Otsu's Thresholding",
          'Gaussian filtered Image','Histogram',"Otsu's Thresholding"]
for i in xrange(3):
    plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
    plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)
    plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
    plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
plt.show()
```

![image-20210104160854801](C:\Users\元宵\AppData\Roaming\Typora\typora-user-images\image-20210104160854801.png)

![image-20210104160937873](C:\Users\元宵\AppData\Roaming\Typora\typora-user-images\image-20210104160937873.png)

[^总结一下]: 简单阈值的阈值固定，自适应阈值根据像素周边像素块的情况确定，大津二值化根据全图像直方图来选取全局阈值。

###### 3.4 大津二值化算法的python实现

略。

---



##### 4. 平滑图像

###### 4.1 2D卷积（过滤）- averaging

```python
# cv.filter2D(原图像，输出图像，卷积核，核中心[默认(-1,-1)])
# 核中心对准原图像的像素
img = cv.imread('opencv_logo.png')
# 使用5*5的卷积核，将核内像素加和/25求平均，替换锚点的像素值
kernel = np.ones((5,5),np.float32)/25
dst = cv.filter2D(img,-1,kernel)
plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(dst),plt.title('Averaging')
plt.xticks([]), plt.yticks([])
plt.show()
```

###### 4.2 图像平滑模糊

```python
img = cv.imread('opencv-logo-white.png')
# Blur-5*5卷积核内，加和平均
blur = cv.blur(img,(5,5))
# Gaussian-5*5卷积核，标准差为0，加权平均
blur2 = cv.GaussianBlur(img,(5,5),0)
# median-取内核区域内的所有像素中值，卷积核Ksize=5
median = cv.medianBlur(img,5)
# bilateralFilter双边过滤器，两个强度不同的高斯滤波器，保证只模糊中央像素，不模糊边缘像。
# cv.bilateralFilter(Input,Output,d,sigmaColor,sigmaSpace)
# 滤波期间使用的像素邻域直径d=9，颜色空间过滤器的sigmaColor=75, 坐标空间中滤波器的sigmaSpace=75
blur3 = cv.bilateralFilter(img,9,75,75)
plt.subplot(233),plt.imshow(blur),plt.title('Blurred')
plt.xticks([]), plt.yticks([])
plt.subplot(234),plt.imshow(blur2),plt.title('GaussianBlur')
plt.xticks([]), plt.yticks([])
plt.subplot(235),plt.imshow(median),plt.title('medianBlur')
plt.xticks([]), plt.yticks([])
plt.subplot(236),plt.imshow(blur3),plt.title('bilateralFilter')
plt.xticks([]), plt.yticks([])
plt.show()
```

![image-20210104163208171](C:\Users\元宵\AppData\Roaming\Typora\typora-user-images\image-20210104163208171.png)

---



##### 5. 形态转换

###### 5.1 形态变换

形态变换是一些基于图像形状的简单操作。包括：侵蚀、膨胀、open(先侵蚀再膨胀)、close(先膨胀再侵蚀)、梯度、tophat、blackhat。

```python
img = cv.imread('j.png',0)
# 5*5卷积核
kernel = np.ones((5,5),np.uint8)
# 侵蚀，内核在图像中滑动，当内核中所有像素为1时，原始像素才为1，否则为0.
erosion = cv.erode(img,kernel,iterations = 1)
# 膨胀，与侵蚀相反，内核只要有一个像素为1，则原始像素为1.
dilation = cv.dilate(img,kernel,iterations = 1)
# open，先侵蚀再膨胀，用于消除噪音
opening = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
# close，先膨胀再侵蚀，用于关闭前景对象内部小孔或小黑点
closing = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
# 梯度，取轮廓
gradient = cv.morphologyEx(img, cv.MORPH_GRADIENT, kernel)
# 顶帽，原始图像与进行open之后得到的图像的差，用来分离比邻近点亮一些的斑块
tophat = cv.morphologyEx(img, cv.MORPH_TOPHAT, kernel)
# 黑帽，进行close以后得到的图像与原图像的差，用来分离比邻近点暗一些的斑块
blackhat = cv.morphologyEx(img, cv.MORPH_BLACKHAT, kernel)
```

![image-20210104190256798](C:\Users\元宵\AppData\Roaming\Typora\typora-user-images\image-20210104190256798.png)

| opening                                                    | closing                                                |
| ---------------------------------------------------------- | ------------------------------------------------------ |
| ![Opening.png](https://docs.opencv.org/3.4.10/opening.png) | ![png.png](https://docs.opencv.org/3.4.10/closing.png) |

###### 5.2 获取内核

```python
# 使用cv.getStructuringElement(内核形状，大小)，即可获得所需的内核。
# 矩形内核
>>> cv.getStructuringElement(cv.MORPH_RECT,(5,5))
array([[1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1]], dtype=uint8)
# 椭圆形内核
>>> cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
array([[0, 0, 1, 0, 0],
       [1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1],
       [0, 0, 1, 0, 0]], dtype=uint8)
# 十字形内核
>>> cv.getStructuringElement(cv.MORPH_CROSS,(5,5))
array([[0, 0, 1, 0, 0],
       [0, 0, 1, 0, 0],
       [1, 1, 1, 1, 1],
       [0, 0, 1, 0, 0],
       [0, 0, 1, 0, 0]], dtype=uint8)
```

---



##### 6. 图像渐变

###### 6.1 边缘检测

```python
img = cv.imread('dave.jpg',0)
# cv.Laplacian(输入图像，输出图像，输出图像深度，核大小)，求二阶导数
laplacian = cv.Laplacian(img,cv.CV_64F)
# cv.soble(输入图像，输出图像，x方向差分阶数，y方向差分阶数，Soble核大小)
# 利用图像边缘灰度值变化率检测图像边缘，soble开启一个3*3的窗口对x和y方向进行分别求导，若对x求导，则某点导数为(第三列之和)-(第一列的之和)
sobelx = cv.Sobel(img,cv.CV_64F,1,0,ksize=5)
sobely = cv.Sobel(img,cv.CV_64F,0,1,ksize=5)
plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
plt.show()
```

![img](https://docs.opencv.org/3.4.10/gradients.jpg)

###### 6.2 注意数据格式转换

```python
# 将输出数据类型保留为更高的形式(cv.CV_16S，cv.CV_64F)，可一次检测出更多边缘，ps,黑色到白色的过渡被视为正斜率(具有正值)，而白色到黑色的过渡被视为负斜率(具有负值)，cv.CV_8U会丢失负斜率边缘
img = cv.imread('box.png',0)
# 输出格式为cv.CV_8U
sobelx8u = cv.Sobel(img,cv.CV_8U,1,0,ksize=5)
# 输出格式为cv.CV_64F，然后取绝对值，再转换回cv.CV_8U
sobelx64f = cv.Sobel(img,cv.CV_64F,1,0,ksize=5)
abs_sobel64f = np.absolute(sobelx64f)
sobel_8u = np.uint8(abs_sobel64f)
plt.subplot(1,3,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(1,3,2),plt.imshow(sobelx8u,cmap = 'gray')
plt.title('Sobel CV_8U'), plt.xticks([]), plt.yticks([])
plt.subplot(1,3,3),plt.imshow(sobel_8u,cmap = 'gray')
plt.title('Sobel abs(CV_64F)'), plt.xticks([]), plt.yticks([])
plt.show()
```

![double_edge.jpg](https://docs.opencv.org/3.4.10/double_edge.jpg)

---



##### 7. 坎尼边缘检测

```python
img = cv.imread('messi5.jpg',0)
# cv.Canny(输入图像，minVal,maxVal,Sobel算子内核大小)
# maxVal决定目标与背景对比度,minVal用于平滑边缘的轮廓
# 注意，cv.Canny只接受单通道图像作为输入
edges = cv.Canny(img,100,200)
plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()
```

![canny1.jpg](https://docs.opencv.org/3.4.10/canny1.jpg)

---



##### 8. 图像金字塔

###### 8.1 图像金字塔的生成

![img](https://img-blog.csdn.net/20180809151348612?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3podV9ob25namk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

多分辨率采样图像，金字塔层级越高，图像越小，分辨率越低。

常见的两类金字塔，高斯金字塔(向下降采样)，拉普拉斯laplacian金字塔(向上采样)

拉普拉斯金字塔仅限边缘图像，只要用于图像压缩，大多数元素为0

![lap.jpg](https://docs.opencv.org/3.4.10/lap.jpg)

```python
img = cv.imread('messi5.jpg')
# 向下采样，图像尺寸减半
lower_reso = cv.pyrDown(img)
lower_reso2 = cv.pyrDown(lower_reso)
lower_reso3 = cv.pyrDown(lower_reso2)
# 向上采样，图片尺寸扩大
higher_reso2 = cv.pyrUp(lower_reso3)
higher_reso3 = cv.pyrUp(higher_reso2)
higher_reso4 = cv.pyrUp(higher_reso3)
# 注意pyrDown和pyrUp并非逆操作，且两种处理均为非线性，不可逆且有损
```

![image-20210104204457685](C:\Users\元宵\AppData\Roaming\Typora\typora-user-images\image-20210104204457685.png)

![image-20210104204711771](C:\Users\元宵\AppData\Roaming\Typora\typora-user-images\image-20210104204711771.png)

###### 8.2 使用金字塔进行图像融合

```python
# 图像拼接中，使用金字塔进行图像融合可以无缝融合图像
A = cv.imread('apple.jpg')
B = cv.imread('orange.jpg')
# 为A生成6层高斯金字塔放在数组gpA中
G = A.copy()
gpA = [G]
for i in xrange(6):
    G = cv.pyrDown(G)
    gpA.append(G)
# 为B生成6层高斯金字塔放在数组gpB中
G = B.copy()
gpB = [G]
for i in xrange(6):
    G = cv.pyrDown(G)
    gpB.append(G)
# 在高斯金字塔基础上，为A生成拉普拉斯金字塔
lpA = [gpA[5]]
for i in xrange(5,0,-1):
    GE = cv.pyrUp(gpA[i])
    L = cv.subtract(gpA[i-1],GE)
    lpA.append(L)
# 在高斯金字塔基础上，为B生成拉普拉斯金字塔
lpB = [gpB[5]]
for i in xrange(5,0,-1):
    GE = cv.pyrUp(gpB[i])
    # 与前一张金字塔图像相减
    L = cv.subtract(gpB[i-1],GE)
    lpB.append(L)
# 每个级别中添加左右各一半的图像
LS = []
for la,lb in zip(lpA,lpB):
    rows,cols,dpt = la.shape
    # np.hstack()矩阵进行行连接
    ls = np.hstack((la[:,0:cols/2], lb[:,cols/2:]))
    LS.append(ls)
# 重建图像
ls_ = LS[0]
for i in xrange(1,6):
    ls_ = cv.pyrUp(ls_)
    ls_ = cv.add(ls_, LS[i])
# 图像各半边直接连接
real = np.hstack((A[:,:cols/2],B[:,cols/2:]))
cv.imwrite('Pyramid_blending2.jpg',ls_)
cv.imwrite('Direct_blending.jpg',real)
```

![orapple.jpg](https://docs.opencv.org/3.4.10/orapple.jpg)

---



##### 9.  轮廓

###### 9.1 轮廓入门（寻找/绘制）

```python
# cv.findContours(原图像，轮廓检索模式，轮廓逼近方法)，返回修改后的图像、轮廓、层次
# 其中轮廓检索模式:
# 	cv::RETR_EXTERNAL：表示只提取最外面的轮廓;
# 	cv::RETR_LIST：表示提取所有轮廓并将其放入列表;
# 	cv::RETR_CCOMP:表示提取所有轮廓并将组织成一个两层结构，其中顶层轮廓是外部轮廓，第二层轮廓是“洞”的轮廓;
# 	cv::RETR_TREE：表示提取所有轮廓并组织成轮廓嵌套的完整层级结构
# 其中轮廓逼近方法：
# 	cv::CHAIN_APPROX_NONE：存储形状边界的所有点,如一条直线轮廓的所有点;
# 	cv::CHAIN_APPROX_SIMPLE：它删除所有冗余点并压缩轮廓，从而节省内存，如一条直线轮廓仅存储为两个点即可;
im = cv.imread('test.jpg')
# 将图片转化成灰度图
imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
# 全局简单阈值将图片从灰度图转为二值图像，ret=127,thresh为处理后的图像
ret, thresh = cv.threshold(imgray, 127, 255, 0)
# im2未返回
im2, contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
# cv.drawContours(原图像，轮廓python列表，轮廓索引，颜色厚度等)
# 注，若要绘制所有轮廓，则轮廓索引传-1
cv.drawContours(im, contours, -1, (0,255,0), 3) #绘制所有轮廓
cv.drawContours(im, contours, 3, (0,255,0), 3) #绘制第3个轮廓
cnt = contours[4]
cv.drawContours(im, [cnt], 0, (0,255,0), 3)
```

![image-20210105121548126](C:\Users\元宵\AppData\Roaming\Typora\typora-user-images\image-20210105121548126.png)

![img 2021_1_5 12_16_42](C:\Users\元宵\Videos\Captures\img 2021_1_5 12_16_42.png)

###### 9.2 轮廓特征

```python
# cv.moments(轮廓)计算图像的中心矩，用于提取轮廓相关信息
img = cv.imread('star.jpg',0)
ret,thresh = cv.threshold(img,127,255,0)
contours,hierarchy = cv.findContours(thresh, 1, 2)
cnt = contours[0]

M = cv.moments(cnt)
print( M )
# 由矩得到质心
cx = int(M['m10']/M['m00'])
cy = int(M['m01']/M['m00'])
# 轮廓区域面积--cv.contourArea
area = cv.contourArea(cnt) = M['m00']
# 轮廓区域周长
# cv.arcLength(轮廓，指定形状为闭合轮廓[true]或曲线)
perimeter = cv.arcLength(cnt,True)
# 轮廓近似
# cv.approxPolyDP(轮廓，轮廓到近似轮廓的最大距离，指定曲线是否闭合)
epsilon = 0.1*cv.arcLength(cnt,True)
approx = cv.approxPolyDP(cnt,epsilon,True)
```

```python
# 求凸包的算法[Sklansky, J., Finding the Convex Hull of a Simple Polygon. PRL 1 $number, pp 79-83 (1982)]
# 求凸包cv.convexHull(凸包点集，输出的凸包点，bool凸包顺时针/逆时针)
hull = cv.convexHull(points[, hull[, clockwise[, returnPoints]]   hull = cv.convexHull(cnt)
```

<img src="https://images0.cnblogs.com/blog/361409/201311/13211312-5a1aec5e4c9749fca1a652fc57be9412.png" alt="image" style="zoom: 80%;" /><img src="https://images0.cnblogs.com/blog/361409/201311/13211326-49718d0b241f4cee85f443713171e23e.png" alt="image" style="zoom: 80%;" />

```python
# 检查曲线是否为凸形，返回true/false
k = cv.isContourConvex(cnt)

# 边界矩形
# 直边角矩形boundingRect，不考虑旋转，面积非最小，返回矩形左上角坐标(x,y)以及矩形的宽高(w,h)
x,y,w,h = cv.boundingRect(cnt)
cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
# 获取最小外接矩阵，中心点坐标，宽高，旋转角度
rect = cv2.minAreaRect(points)
# 获取矩形四个顶点，浮点型
box = cv2.boxPoints(rect)
# 取整
box = np.int0(box)
# 画出轮廓
cv.drawContours(img,[box],0,(0,0,255),2)

# 外接圆
(x,y),radius = cv.minEnclosingCircle(cnt)
center = (int(x),int(y))
radius = int(radius)
# 画圆 cv.circle(原图像，中心点坐标，半径，颜色，粗细)
cv.circle(img,center,radius,(0,255,0),2)

# 拟合椭圆
# cv.fitEllipse()返回RotatedRect类型的值，通过一系列数学公式可以从RotatedRect获取到椭圆参数
ellipse = cv.fitEllipse(cnt)
# 画椭圆 cv.ellipse(原图像，椭圆参数，颜色，粗细)
cv.ellipse(img,ellipse,(0,255,0),2)

# 直线拟合
rows,cols = img.shape[:2]
# cv.fitLine(二维点数组，输出直线，距离类型，距离参数[0]，径向精度参数[1e-2]，角度精度参数[1e-2])，
[vx,vy,x,y] = cv.fitLine(cnt, cv.DIST_L2,0,0.01,0.01)
lefty = int((-x*vy/vx) + y)
righty = int(((cols-x)*vy/vx)+y)
# cv.line(原图像，直线起点，直线终点，颜色，粗细)
cv.line(img,(cols-1,righty),(0,lefty),(0,255,0),2)
```

<img src="https://docs.opencv.org/3.4.10/boundingrect.png" alt="boundingrect.png" style="zoom: 67%;" /><img src="https://docs.opencv.org/3.4.10/circumcircle.png" alt="circumcircle.png" style="zoom: 67%;" /><img src="https://docs.opencv.org/3.4.10/fitline.jpg" alt="fitline.jpg" style="zoom: 80%;" />

<img src="C:\Users\元宵\AppData\Roaming\Typora\typora-user-images\image-20210105134827585.png" alt="image-20210105134827585" style="zoom: 50%;" />

<img src="https://img2018.cnblogs.com/blog/1113434/201812/1113434-20181204102311664-917625192.png" alt="img" style="zoom:80%;" />

###### 9.3 轮廓属性

```python
# 获取对象边界矩形的左上点坐标(x,y)和矩阵的宽高(w,h), 长宽比 aspect_ratio
x,y,w,h = cv.boundingRect(cnt)
aspect_ratio = float(w)/h

# 范围 = 轮廓面积/边界矩形区域面积
area = cv.contourArea(cnt)
x,y,w,h = cv.boundingRect(cnt)
rect_area = w*h
extent = float(area)/rect_area

# 坚固性 = 轮廓面积/凸包面积
area = cv.contourArea(cnt)
hull = cv.convexHull(cnt)
hull_area = cv.contourArea(hull)
solidity = float(area)/hull_area

# 等效直径 = 与轮廓面积等值的圆的直径
area = cv.contourArea(cnt)
equi_diameter = np.sqrt(4*area/np.pi)

# 方向，给出长轴长度MA和短轴长度ma
(x,y),(MA,ma),angle = cv.fitEllipse(cnt)

# 遮罩
mask = np.zeros(imgray.shape,np.uint8) #生成全0像素值的图像形状的遮罩
cv.drawContours(mask,[cnt],0,255,-1) #在遮罩上画出全部轮廓(白色)
pixelpoints = np.transpose(np.nonzero(mask)) #返回结果是构成图像的所有像素点
#pixelpoints = cv.findNonZero(mask)

# 使用遮罩图像可以得到最大值最小值及其位置
min_val, max_val, min_loc, max_loc = cv.minMaxLoc(imgray,mask = mask)

# 平均颜色和平均灰度
mean_val = cv.mean(im,mask = mask)

# 极点，对象最上/下/左/右边的点
leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])
```

<img src="C:\Users\元宵\AppData\Roaming\Typora\typora-user-images\image-20210105145651318.png" alt="image-20210105145651318" style="zoom: 50%;" /><img src="https://gimg2.baidu.com/image_search/src=http%3A%2F%2F07.imgmini.eastday.com%2Fmobile%2F20170111%2F20170111143444_f2b5cb0d109a3f39eefc0d3005c58914_3.jpeg&refer=http%3A%2F%2F07.imgmini.eastday.com&app=2002&size=f9999,10000&q=a80&n=0&g=0n&fmt=jpeg?sec=1612421691&t=480f839413bd0ed68c427539f14fe343" alt="img" style="zoom: 33%;" />

###### 9.4 轮廓其他功能



```python
img = cv.imread('star.jpg')
img_gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY) #转为灰度图
ret,thresh = cv.threshold(img_gray, 127, 255,0) #变成二值图
contours,hierarchy = cv.findContours(thresh,2,1) #提取轮廓
cnt = contours[0]
# 
hull = cv.convexHull(cnt,returnPoints = False)
defects = cv.convexityDefects(cnt,hull)
for i in range(defects.shape[0]):
    s,e,f,d = defects[i,0]
    start = tuple(cnt[s][0])
    end = tuple(cnt[e][0])
    far = tuple(cnt[f][0])
    cv.line(img,start,end,[0,255,0],2)
    cv.circle(img,far,5,[0,0,255],-1)
cv.imshow('img',img)
cv.waitKey(0)
cv.destroyAllWindows()
```



```python
dist = cv.pointPolygonTest(cnt,(50,50),True)
img1 = cv.imread('star.jpg',0)
img2 = cv.imread('star2.jpg',0)
ret, thresh = cv.threshold(img1, 127, 255,0)
ret, thresh2 = cv.threshold(img2, 127, 255,0)
im2,contours,hierarchy = cv.findContours(thresh,2,1)
cnt1 = contours[0]
im2,contours,hierarchy = cv.findContours(thresh2,2,1)
cnt2 = contours[0]
ret = cv.matchShapes(cnt1,cnt2,1,0.0)
print( ret )
```


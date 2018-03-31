#Practice: 載入一張圖片，並將其RGB變成一半在輸出

'''
### (1)透過getpixel / putpixel API來直接些改圖片的RGB值
from PIL import Image
img = Image.open("123.jpg")
print (img.format,img.size,img.mode)
img.show()
print(img.size[0],img.size[1])
for x in range(img.size[0]):
    for y in range(img.size[1]):
        r,g,b = img.getpixel((x,y))
        img.putpixel((x,y),( int(r/2),int(g/2),int(b/2) ))
img.save("temp.jpg")
img.show()
'''

### (2)透過load()將圖片載入到記憶體
### load()會將圖片載入到記憶體，並回傳一個Pixel Access Object
### 透過載入圖片到內存，使得操作比起putpixel / getpixel 來的快很多
from PIL import Image

img = Image.open("123.jpg")
img.show()
pixel = img.load()
for x in range(img.size[0]):
    for y in range(img.size[1]):
        r,g,b = pixel[x,y]
        #print(r,g,b)
        pixel[x,y] = (int(r/2),int(g/2),int(b/2))
        #print(pixel[x,y])
img.save("temp2.jpg")
img.show()

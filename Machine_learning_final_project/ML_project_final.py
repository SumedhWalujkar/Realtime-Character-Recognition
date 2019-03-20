def datasetloader():
    import numpy as np
    import csv
    import matplotlib.pyplot as plt
    import numpy as np
    link="C:\\Users\\Sumedh Walujkar\\Desktop\\final_ml\\dataset\\emnist-letters-train.csv"
    inputfile=open(link,"r")    
    y=[]
    for line in inputfile:
         y.append(line.split(",")) 
    y=datamodeller(y)
    return y
##############################################################################################33
def datamodeller(y):
    for iter in range(0,len(y)):
        y[iter] = list(map(int,y[iter])) #converting the string values to int
    return y

def labels_and_attributes(arr):
    import numpy as np
    xx=[]
    yy=[]
    
    for i in range (1,len(arr)):
      
                xx.append(arr[i][1:])
                yy.append(arr[i][0])
    tem=(np.zeros(784,))
    xx.append(tem)
    yy.append(0)
    
    return xx,yy
####################################################################################################3
def  image_loader(k):
    import numpy as np
    import csv
    import matplotlib.pyplot as plt
    from PIL import Image
    '''
    please just change the "C:/Users/Sumedh Walujkar/Desktop/images" part
   '''
    image_file = Image.open("C:/Users/Sumedh Walujkar/Desktop/images"+"/"+str(k)+".png")
    width,height=image_file.size
    temp=int(round((width/height),0))
    #temp=6
    width/height
    new_height = 28
    new_width  = new_height* temp
    #new_width = new_width/height
    img = image_file.resize((int(new_width),new_height), Image.ANTIALIAS)
    img=np.asarray(img)
    imager22=255-img[:][:]
    c=[]
    for k in range(0,temp):
        b=[]
        for i in range(0,28):
            a=[]
            for j in range(0,28):
                a.append(imager22[i][(28*k)+j])
            b.append(a)
        c.append(b)
    c=np.asarray(c)
    x=[]
    for iter in range(0,len(c)):
        c[iter]=np.transpose(c[iter])
        x.append(c[iter].reshape(784,))  
        
    return image_file,imager22,x,temp
#################################################################################################3
def predictor(k):
    image_file,whole_image,x,number_of_elements=image_loader(str(k))
    x=np.asarray(x)
    X_test = x.reshape(number_of_elements, 28, 28, 1).astype('float32')
    result=model.predict_classes(X_test)
    output=[]
    for iter in range(0,len(result)):
        output.append(chr(result[iter]+64))
    plt.imshow(image_file, cmap='gray')
    plt.show() 
    print("The words are")
    print(output)
    
        
    
##################################################################################################
print("please change the link to the 'emnist-letters-train.csv' in the method dataloader() and link of the folder images in the method image-loader()")
import numpy as np
import csv
import matplotlib.pyplot as plt
from PIL import Image 
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Flatten, Conv2D, MaxPooling2D
dataset = datasetloader()
type(dataset[0][0])

TrainX, TrainY = labels_and_attributes(dataset)
 



xx=np.asarray(TrainX)

X_train = xx.reshape(88800, 28, 28, 1).astype('float32')

n_classes = 27
Y_train = keras.utils.to_categorical(TrainY, n_classes)

model = Sequential()
model.add(Conv2D(32,kernel_size=(3,3), activation='relu', input_shape=(28,28,1)))
model.add(Conv2D(64,kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(n_classes, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
model.fit(X_train, Y_train,batch_size=128,epochs=2,verbose=1)


for i in range(1,9):
    predictor(i)
    

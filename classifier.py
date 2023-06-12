import tensorflow as tf
import os
import cv2
import imghdr  #used to check img extension
from matplotlib import pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Dense,Flatten
from tensorflow.keras.metrics import Precision,Recall,BinaryAccuracy
import tkinter
from tkinter import *
from tkinter.ttk import *


data_dir='image4classify'
#print(os.listdir(os.path.join(data_dir,'/content/drive/MyDrive/images4classify/humans')))
img_exts=['jpg','png','jpeg','bmp']

img=cv2.imread(os.path.join('image4classify','human','human/22selfie2.jpg'))
#type(img)
#img.shape
#plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
#img.show()


for image_class in os.listdir(data_dir):
  for image in os.listdir(os.path.join(data_dir,image_class)):
    image_path=os.path.join(data_dir,image_class,image)
    try:
      img=cv2.imread(image_path)
      tip=imghdr.what(image_path)
      if tip not in img_exts:
        print('image not in ext list {} '.format(image_path))
        os.remove(image_path)
    except Exception as e:
        print('issue with image {}'.format(image_path))


data=tf.keras.utils.image_dataset_from_directory('image4classify')#building data pipeline


#scaled=batch[0]/255 to range b/w 0.0 to 1.0
#scaled.max() eqls 1.0

#----------------------
#preprocessing the data

#1.scaling the data
data=data.map(lambda x,y: (x/255,y))
scaled_iterator=data.as_numpy_iterator()
batch=scaled_iterator.next()
#print(batch[1])


 #to view 0 or 1 to human or object
fig,ax=plt.subplots(ncols=4,figsize=(20,20))
for idx,img in enumerate(batch[0][:4]):     #1 for objects,0 for people
  ax[idx].imshow(img)
  ax[idx].title.set_text(batch[1][idx])

#2.splitting data

train_size=int(len(data)*.7)
val_size=int(len(data)*.2)
test_size=int(len(data)*.1)+1


print(len(data))
print(train_size)
print(val_size)
print(test_size)

train=data.take(train_size)
val=data.skip(train_size).take(val_size)
test=data.skip(train_size+val_size).take(test_size)



#training

model=Sequential()

model.add(Conv2D(16,(3,3),1,activation='relu',input_shape=(256,256,3)))
model.add(MaxPooling2D())

model.add(Conv2D(32,(3,3),1,activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(16,(3,3),1,activation='relu'))
model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(256,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer='adam',loss=tf.losses.BinaryCrossentropy(),metrics=['accuracy'])

model.summary()

logdir='logs'
tensorboard_callback=tf.keras.callbacks.TensorBoard(log_dir=logdir)
hist=model.fit(train, epochs=2, validation_data=val, callbacks=[tensorboard_callback])
#g=tf.keras.utils.plot_model()


#plotting the performance

fig=plt.figure()
plt.plot(hist.history['loss'],color='teal',label='loss')
plt.plot(hist.history['val_loss'],color='orange',label='val_loss')
fig.suptitle('loss',fontsize=20)
plt.legend(loc="upper left")
plt.show()


#evaluation
pre=Precision()
re=Recall()
acc=BinaryAccuracy()

for batch in test.as_numpy_iterator():
  X,y =batch
  yhat = model.predict(X)
  pre.update_state(y,yhat)
  re.update_state(y,yhat)
  acc.update_state(y,yhat)

print(f'Precision:{pre.result().numpy()},Recall:{re.result().numpy()},Accuracy:{acc.result().numpy()}')


frame=Tk()
frame.title("cv")
frame.geometry("500x500")
im = tkinter.StringVar()

def submit():

  name = im.get()
  im.set("")
  img=cv2.imread(name)
  plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


  resize=tf.image.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),(256,256))
  plt.imshow(resize.numpy().astype(int))
  plt.show()

  yhat=model.predict(np.expand_dims(resize/255,0))
  print(yhat)

  if  yhat >0.5:
    print(f'Predicted class is people')
    result = tkinter.Label(frame, text='Predicted class is people', font=('calibre', 10, 'bold'))
    result.pack()
  else:
    print(f'Predicted class is book')
    result = tkinter.Label(frame, text='Predicted class is book', font=('calibre', 10, 'bold'))
    result.pack()




heading=Label(frame,text="IMAGE CLASSIFICATION") #print heading
heading.pack()
name_label = tkinter.Label(frame, text='enter image', font=('calibre', 10, 'bold'))
name_label.pack()
name_entry = tkinter.Entry(frame,textvariable =im, font=('calibre',10,'normal'))
name_entry.pack()
sub_btn=tkinter.Button(frame,text = 'Submit', command = submit)
sub_btn.pack()
qbtn=Button(frame,text="quit",command=frame.destroy)
qbtn.pack()
mainloop()

#Common 4 steps for fine tuning#

1. set input shape
2. get the last layer output
3. do some convolution network if you need to
4. use GlobalAveragePooling   or Flatten()  to prepare for linking later (your customized) fc networks
5. build your own fcs


```python
# set your own input shape
your_own_input_shape = (400,400,3)
input_tensor = Input(input_shape =your_own_input_shape) 

basemodel = VGG16(weights='imagenet', include_top=False, input_tensor=input_tensor)

# get the output of the pretrained model
x = base_model.output   # same to   x=basemodel.layers[-1].output

# use GlobalAvgPool to move Conv to vectors
x = GlobalAveragePooling2D()(x)   # using GlobalAverage to avoid Flatten()

#build your own fc 
x = Dense(units=512, activation='relu', name='fc1')(x)
x = Dropout(rate=0.5)(x)
x = Dense(units=512, activation='relu', name='fc1')(x)
x = Dropout(rate=0.5)(x)
x = Dense(units=num_classes, activation='softmax', name='fc1')(x)


```

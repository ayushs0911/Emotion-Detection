## Emotion-Detection
**Significance of Analysis **
- Malaria is a life-threatening disease caused by parasites that are transmitted to people through the bites of infected female Anopheles mosquitoes. It is preventable and curable.
- In 2021, there were an estimated 247 million cases of malaria worldwide.
- The estimated number of malaria deaths stood at 619 000 in 2021.

**Dataset** <br>
```
dataset, dataset_info = tfds.load('malaria',
                                  with_info = True,
                                  as_supervised = True, 
                                  shuffle_files = True,
                                  split = ['train'])
```

**Data Augmentation **<br>
Mixup, CutNix and Albumentations

**Feature Extraction** <br>
```

feature_extractor_seq_model = tf.keras.Sequential([
                             InputLayer(input_shape = (IM_SIZE, IM_SIZE, 3)),

                             Conv2D(filters = 6, kernel_size = 3, strides=1, padding='valid', activation = 'relu'),
                             BatchNormalization(),
                             MaxPool2D (pool_size = 2, strides= 2),

                             Conv2D(filters = 16, kernel_size = 3, strides=1, padding='valid', activation = 'relu'),
                             BatchNormalization(),
                             MaxPool2D (pool_size = 2, strides= 2),

                             

])
feature_extractor_seq_model.summary()
```

**Modeling** <br>
```

func_input = Input(shape = (IM_SIZE, IM_SIZE, 3), name = "Input Image")

x = feature_extractor_seq_model(func_input)

x = Flatten()(x)

x = Dense(100, activation = "relu")(x)
x = BatchNormalization()(x)

x = Dense(10, activation = "relu")(x)
x = BatchNormalization()(x)

func_output = Dense(1, activation = "sigmoid")(x)

model_1 = Model(func_input, func_output, name = "Lenet_Model")
model_1.summary()
```




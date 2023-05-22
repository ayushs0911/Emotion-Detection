## Malaria Diagnosis
**Significance of Analysis**
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
**Evaluation Curves** <br>
<img width="400" alt="Screenshot 2023-05-22 at 1 25 50 PM" src="https://github.com/ayushs0911/Emotion-Detection/assets/122048067/0d6aaeb6-4fe2-406d-b80e-08ecce5baf54">
<img width="400" alt="Screenshot 2023-05-22 at 1 25 59 PM" src="https://github.com/ayushs0911/Emotion-Detection/assets/122048067/87bd3b51-acad-4de7-a1d2-93f42ab8ebbb">

## Confusion Maxtrix
<img width="400" alt="Screenshot 2023-05-22 at 1 27 07 PM" src="https://github.com/ayushs0911/Emotion-Detection/assets/122048067/cf3dcccf-6340-4719-9e60-28ead88f1232">
<br>

## ROC CURVE ANALYSIS<br>

<img width="400" alt="Screenshot 2023-05-22 at 1 27 19 PM" src="https://github.com/ayushs0911/Emotion-Detection/assets/122048067/feecad7f-9025-4f99-9009-c6f38a59ba6b">
<br>

## Replotted Confusion Matrix according to new Threshold<br>

<img width="400" alt="Screenshot 2023-05-22 at 1 27 35 PM" src="https://github.com/ayushs0911/Emotion-Detection/assets/122048067/dd5eb97d-43f0-4c2e-a0a7-b94bdb901957">

import pandas as pd
import cv2
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from ydata_profiling import ProfileReport

from keras.api.models import Model
from keras.api.layers import Dropout, Dense, BatchNormalization
from keras.api.applications.mobilenet_v2 import MobileNetV2
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.src.callbacks import ModelCheckpoint


class Dataset:
    def __init__(self,
                 data_path):
        
        # Setting up class members
        self.data_path = data_path
        self.list_attr_celeba_path = f"{data_path}/list_attr_celeba.csv"
        self.list_bbox_celeba_path = f"{data_path}/list_bbox_celeba.csv"
        self.list_eval_partition_path = f"{data_path}/list_eval_partition.csv"
        self.list_landmarks_align_celeba_path = f"{data_path}/list_landmarks_align_celeba.csv"
        self.images_path = f"{data_path}/img_align_celeba/img_align_celeba/"

        # Reading data into pandas dataframes
        self.list_attr_celeba_df = pd.read_csv(self.list_attr_celeba_path)
        self.list_bbox_celeba_df = pd.read_csv(self.list_bbox_celeba_path)
        self.list_eval_partition_df = pd.read_csv(self.list_eval_partition_path)
        self.list_landmarks_align_celeba_df = pd.read_csv(self.list_landmarks_align_celeba_path)

        # Dataset main dataframe
        temp_df = self.list_attr_celeba_df.merge(self.list_bbox_celeba_df, on='image_id')
        temp_df = temp_df.merge(self.list_bbox_celeba_df, on='image_id')  # Should this be done here? If ever?
        temp_df = temp_df.merge(self.list_eval_partition_df, on='image_id')
        self.df = temp_df.merge(self.list_landmarks_align_celeba_df, on='image_id')

    def genereate_report(self, report_name: str = 'Your_Report') -> None:
        profile = ProfileReport(self.df, title="Pandas Profiling Report")
        profile.to_file(f"{report_name}.html")

    def show_some_examples(self, attribute: str, positive: bool = True) -> None:
        """
        Quick visualization of some examples

        :param attribute: One of boolean attribute:
        '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose',
        'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows','Chubby', 'Double_Chin', 'Eyeglasses',
        'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache',
        'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks',
        'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick',
        'Wearing_Necklace', 'Wearing_Necktie', 'Young'
        :type attribute: str
        :param positive: show positive examples or not (negative if false) <-- meaning that if positive then this image has this attribute
        :type positive: boolean
        :return: None
        :rtype: None
        """

        print(self.df.columns)
        if positive:
            exist = 1
        else:
            exist = -1

        df = self.df[self.df[attribute] == exist]
        images = list(df.head(9)['image_id'])
        images = [cv2.imread(self.images_path + path) for path in images]
        horizontal_1 = np.concatenate((images[0], images[1], images[2]), axis=1)
        horizontal_2 = np.concatenate((images[3], images[4], images[5]), axis=1)
        horizontal_3 = np.concatenate((images[6], images[7], images[8]), axis=1)
        vertical = np.concatenate((horizontal_1, horizontal_2, horizontal_3), axis=0)
        cv2.imshow("Images", vertical)

        # Wait for user input
        cv2.waitKey(0)
        cv2.destroyAllWindows()

class CelebA():
  '''
  Wraps the celebA dataset, allowing an easy way to:
    - Select the features of interest,
    - Split the dataset into 'training', 'test' or 'validation' partition.
  '''
  def __init__(self, main_folder='kaggle-celeba-dataset/archive/', selected_features=None, drop_features=[]):
    self.main_folder = main_folder
    self.images_folder   = os.path.join(main_folder, 'img_align_celeba/img_align_celeba/')
    self.attributes_path = os.path.join(main_folder, 'list_attr_celeba.csv')
    self.partition_path  = os.path.join(main_folder, 'list_eval_partition.csv')
    self.selected_features = selected_features
    self.features_name = []
    self.__prepare(drop_features)

  def __prepare(self, drop_features):
    '''do some preprocessing before using the data: e.g. feature selection'''
    # attributes:
    if self.selected_features is None:
      self.attributes = pd.read_csv(self.attributes_path)
      self.num_features = 40
    else:
      self.num_features = len(self.selected_features)
      self.selected_features = self.selected_features.copy()
      self.selected_features.append('image_id')
      self.attributes = pd.read_csv(self.attributes_path)[self.selected_features]

    # remove unwanted features:
    for feature in drop_features:
      if feature in self.attributes:
        self.attributes = self.attributes.drop(feature, axis=1)
        self.num_features -= 1
      
    self.attributes.set_index('image_id', inplace=True)
    self.attributes.replace(to_replace=-1, value=0, inplace=True)
    self.attributes['image_id'] = list(self.attributes.index)
  
    self.features_name = list(self.attributes.columns)[:-1]
  
    # load ideal partitioning:
    self.partition = pd.read_csv(self.partition_path)
    self.partition.set_index('image_id', inplace=True)
  
  def split(self, name='training', drop_zero=False):
    '''Returns the ['training', 'validation', 'test'] split of the dataset'''
    # select partition split:
    if name == 'training':
      to_drop = self.partition.where(lambda x: x != 0).dropna()
    elif name == 'validation':
      to_drop = self.partition.where(lambda x: x != 1).dropna()
    elif name == 'test':  # test
      to_drop = self.partition.where(lambda x: x != 2).dropna()
    else:
      raise ValueError('CelebA.split() => `name` must be one of [training, validation, test]')

    partition = self.partition.drop(index=to_drop.index)
      
    # join attributes with selected partition:
    joint = partition.join(self.attributes, how='inner').drop('partition', axis=1)

    if drop_zero is True:
      # select rows with all zeros values
      return joint.loc[(joint[self.features_name] == 1).any(axis=1)]
    elif 0 <= drop_zero <= 1:
      zero = joint.loc[(joint[self.features_name] == 0).all(axis=1)]
      zero = zero.sample(frac=drop_zero)
      return joint.drop(index=zero.index)

    return joint

def build_model(num_features):
    base = MobileNetV2(input_shape=(224, 224, 3),
                    weights=None,
                    include_top=False,
                    pooling='avg')  # GlobalAveragePooling 2D
  
    # model top
    x = base.output
    x = Dense(1536, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    top = Dense(num_features, activation='sigmoid')(x)
    
    return Model(inputs=base.input, outputs=top)


if __name__ == '__main__':
    dataset = Dataset("kaggle-celeba-dataset/archive/")
    dataset.show_some_examples(attribute="No_Beard")

    celeba = CelebA()
    print(celeba.attributes.sample(5))

    print(celeba.num_features)
    model = build_model(celeba.num_features)
    model.summary()

    # augumentations for training set:
    train_datagen = ImageDataGenerator(rotation_range=20, 
                                    rescale=1./255, 
                                    width_shift_range=0.2, 
                                    height_shift_range=0.2, 
                                    shear_range=0.2, 
                                    zoom_range=0.2, 
                                    horizontal_flip=True, 
                                    fill_mode='nearest')

    # only rescaling the validation set
    valid_datagen = ImageDataGenerator(rescale=1./255)


    # get training and validation set:
    train_split = celeba.split('training'  , drop_zero=False)
    valid_split = celeba.split('validation', drop_zero=False)

    batch_size = 80
    num_epochs = 10

    # data generators:
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_split,
        directory=celeba.images_folder,
        x_col='image_id',
        y_col=celeba.features_name,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='other',
        validate_filenames=False
    )

    valid_generator = valid_datagen.flow_from_dataframe(
        dataframe=valid_split,
        directory=celeba.images_folder,
        x_col='image_id',
        y_col=celeba.features_name,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='other',
        validate_filenames=False
    )

    model.compile(loss='binary_crossentropy',
                optimizer='adadelta',
                metrics=['binary_accuracy'])

    save_path = 'saved_data'

    # setup checkpoint callback:
    model_path = f"{save_path}/weights-FC{celeba.num_features}-MobileNetV2" \
    + "{val_binary_accuracy:.2f}.keras"

    checkpoint = ModelCheckpoint(
        model_path,
        monitor='val_binary_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1)

    history = model.fit(
        train_generator,
        epochs=num_epochs,
        steps_per_epoch=len(train_generator),
        validation_data=valid_generator,
        validation_steps=len(valid_generator),
        # max_queue_size=1,
        shuffle=True,
        callbacks=[checkpoint],
        verbose=1)

    # get testset, and setup generator:
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_set = celeba.split('test', drop_zero=False)

    test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_set,
        directory=celeba.images_folder,
        x_col='image_id',
        y_col=celeba.features_name,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='other')

    # evaluate model:
    score = model.evaluate_generator(
        test_generator,
        steps=len(test_generator),
        max_queue_size=1,
        verbose=1)

    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
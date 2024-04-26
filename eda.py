import pandas as pd
from pandas_profiling import ProfileReport
import cv2
import numpy as np


class Dataset:
    def __init__(self,
                 list_attr_celeba_path="list_attr_celeba.csv",
                 list_bbox_celeba_path="list_bbox_celeba.csv",
                 list_eval_partition_path="list_eval_partition.csv",
                 list_landmarks_align_celeba_path="list_landmarks_align_celeba.csv",
                 images_path="/home/ragnarlodbrok1992/Desktop/SDA/projekt-cv/img_align_celeba/img_align_celeba/"
                 ):

        list_attr_celeba = pd.read_csv(list_attr_celeba_path)
        list_bbox_celeba = pd.read_csv(list_bbox_celeba_path)
        list_eval_partition = pd.read_csv(list_eval_partition_path)
        list_landmarks_align_celeba = pd.read_csv(list_landmarks_align_celeba_path)

        df = list_attr_celeba.merge(list_bbox_celeba, on='image_id')
        df = df.merge(list_bbox_celeba, on='image_id')
        df = df.merge(list_eval_partition, on='image_id')
        self.df = df.merge(list_landmarks_align_celeba, on='image_id')
        self.images_path = images_path

    def generate_report(self, report_name='your_report'):
        profile = ProfileReport(self.df, title="Pandas Profiling Report")
        profile.to_file(f"{report_name}.html")

    def show_some_examples(self, attribute: str, positive: bool = True):
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
        :param positive: show positive examples or not (negative if false)
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
        images = [cv2.imread(self.images_path+path) for path in images]
        horizontal_1 = np.concatenate((images[0], images[1], images[2]), axis=1)
        horizontal_2 = np.concatenate((images[3], images[4], images[5]), axis=1)
        horizontal_3 = np.concatenate((images[6], images[7], images[8]), axis=1)
        vertical = np.concatenate((horizontal_1, horizontal_2, horizontal_3), axis=0)
        cv2.imshow("Examples", vertical)

        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    dataset = Dataset()
    dataset.show_some_examples(attribute="5_o_Clock_Shadow", positive=False)




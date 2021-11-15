import cv2
import numpy as np
from PIL import Image
from glob import glob

class Make_gif():
    def __init__(self):
        #あらかじめgifを読み込んでおく（現在は3種類）
        self.gif_dict = {}
        self.gif_files = glob("./gif2/*gif")
        self.fps = None
        for e,ALPHA_GIF_PATH in enumerate(self.gif_files):
            gif_image_list = []
            cap2 = cv2.VideoCapture(ALPHA_GIF_PATH) # gifファイルを読み込み
            self.fps = cap2.get(cv2.CAP_PROP_FPS)        # fps取得

            while True:
                ret, frame = cap2.read()
                if not ret:
                    break

                # BGRをRGBにする
                img_array = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # numpyのarrayからPillowのimage objectを作る
                img = Image.fromarray(img_array)
                gif_image_list.append(img)
            self.gif_dict[e] = gif_image_list
        print("gif読み込み完了")

    #平均色からgifを選択する関数
    def select_gif(self, R_ave, G_ave, B_ave):
        num = 0
        #青の閾値(パイナップル)
        if G_ave >= 150:
            num = 2
        # オレンジの閾値（サーモン）
        elif R_ave >= 180 and G_ave <= 110 and B_ave <= 110:
            num = 1
        #角煮
        elif R_ave <= 140 and G_ave <= 120 and B_ave <= 100:
            num = 0
        return num

    #maskやsanti,nitiとgifを合成し，実際に表示する画像(gif)を生成する関数
    def make_gif_image(self,json_np):
        print("gif合成開始")
        im_list = [] # Pillowのimageリスト
        for num,dic in json_np.items():
            sara = dic["皿"]
            base = dic["三値"]
            mask = dic["二値"]
            R_ave,G_ave,B_ave = dic["平均色"]
            #平均色からgif選択
            gif_num = self.select_gif(R_ave=R_ave, G_ave=G_ave, B_ave=B_ave)             
            gif_image_list = self.gif_dict[gif_num]
            #gif合成
            if num==0:
                for img in gif_image_list:
                    img = img.resize(mask.size)
                    base_copy = base.copy()
                    base_copy.paste(img,(0,0),mask=mask)
                    new_image = np.array(base_copy, dtype=np.uint8)
                    im_list.append(new_image)      #合成画像をリスト追加
            else:
                for e,img in enumerate(gif_image_list):
                    img = img.resize(mask.size)           #マスク画像にリサイズ
                    base_copy = base.copy()
                    base_copy.paste(img,(0,0),mask=mask)
                    new_base = Image.fromarray(im_list[e])
                    new_base.paste(base_copy,(0,0),mask=sara)
                    new_image = np.array(new_base, dtype=np.uint8)
                    im_list[e] = new_image      #合成画像をリスト追加
        print("gif合成終了")
        return im_list,self.fps
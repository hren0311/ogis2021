import cv2
import math
import numpy as np
from PIL import Image

class Make_mask():
    def __init__(self,frame):
        self.all_mask_dict = {}
        self.frame = frame
        self.gray = cv2.cvtColor(self.frame, cv2.COLOR_RGB2GRAY)
        self.gray = cv2.GaussianBlur(self.gray, (33,33), 1)
        #白黒に二値化したときに，thresholdより白い部分の色を取得しない
        self.threshold = 150

        #円検出用パラメータ
        # self.min_radius = 30
        self.min_radius = 20
        self.max_radius = 200
        self.radius_num = 2

        #矩形検出用パラメータ
        self.cond_area = 1000

    ###矩形検知用###
    def angle(self, pt1, pt2, pt0) -> float:
        dx1 = float(pt1[0,0] - pt0[0,0])
        dy1 = float(pt1[0,1] - pt0[0,1])
        dx2 = float(pt2[0,0] - pt0[0,0])
        dy2 = float(pt2[0,1] - pt0[0,1])
        v = math.sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) )
        return (dx1*dx2 + dy1*dy2)/ v

    def findSquares(self, bin_image, image):
        # 輪郭取得
        contours,hierarchy = cv2.findContours(bin_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for i, cnt in enumerate(contours):
            # 輪郭の周囲に比例する精度で輪郭を近似する
            arclen = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, arclen*0.02, True)

            #四角形の輪郭は、近似後に4つの頂点があります。
            #比較的広い領域が凸状になります。

            # 凸性の確認 
            area = abs(cv2.contourArea(approx))
            if approx.shape[0] == 4 and area > self.cond_area and cv2.isContourConvex(approx) :
                maxCosine = 0

                for j in range(2, 5):
                    # 辺間の角度の最大コサインを算出
                    cosine = abs(self.angle(approx[j%4], approx[j-2], approx[j-1]))
                    maxCosine = max(maxCosine, cosine)

                # すべての角度の余弦定理が小さい場合
                #（すべての角度は約90度です）次に、quandrangeを書き込みます
                # 結果のシーケンスへの頂点
                if maxCosine < 0.3 :
                    # 四角判定!!
                    rcnt = approx.reshape(-1,2)
                    new_img = np.zeros(image.shape,dtype=np.uint8)
                    cv2.fillPoly(new_img, [rcnt],(255,255,255), lineType=cv2.LINE_8, shift=0)
                    new_img = cv2.cvtColor(new_img, cv2.COLOR_RGB2GRAY)
            
                    return new_img
    ###矩形検知用終了###

    #frameから円検出を行い，マスク作成
    def detection_circle(self):
        mask_list = []
        circles = cv2.HoughCircles(self.gray, cv2.HOUGH_GRADIENT, 1, 60, param1=10, param2=85, minRadius=self.min_radius, maxRadius=self.max_radius)
        if str(circles) != "None":
            circles = np.uint16(np.around(circles))
            for e,i in enumerate(circles[0,:]):

                mask = np.zeros((self.frame.shape[0],self.frame.shape[1]),dtype=np.uint8)
                cv2.circle(mask,(i[0],i[1]),0,(255,255,255),i[2]*2-20)
                mask_list.append(mask)

                #皿検知の上限を定めて置く．現在は2枚
                if e+1==self.radius_num:
                    break

        return mask_list

    #frameから矩形検出を行い，マスク作成
    def detection_rect(self):
        mask_list = []
        _, bw = cv2.threshold(self.gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        mask = self.findSquares(bw, self.frame)
        if mask != None:
            mask_list.append(mask)
        return mask_list

    #マスクから皿領域内の料理部分だけの三値画像と二値画像を作成する
    def make_santi_and_niti(self,mask):
        #マスクと同じサイズの白画像(white)に皿(mask)をマスクにし，
        #フレームを貼り付ける➡皿内部以外の領域が白の画像
        white = np.ones((mask.shape[0],mask.shape[1],3),dtype=np.uint8)*255
        white_pil = Image.fromarray(white)
        frame_pil = Image.fromarray(self.frame)
        mask_pil = Image.fromarray(mask)
        white_pil.paste(frame_pil,(0,0),mask=mask_pil)

        
        crop_gray = cv2.cvtColor(np.array(white_pil), cv2.COLOR_RGB2GRAY)
        #thresholdを使い，料理座標を取得
        rr,cc = np.where(crop_gray<=self.threshold)
        #平均色取得
        average_color = self.calc_average_color(white=white,rr=rr,cc=cc)

        #三値用マスク作成(料理部分だけ白，それ以外は黒の画像)
        santi_mask = np.zeros((mask.shape[0],mask.shape[1]),dtype=np.uint8)
        santi_mask[rr,cc] = 255

        #三値作成(皿マスクをRGBに変換し，料理部分のみframeから貼り付ける)
        santi_pil = Image.fromarray(cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB))
        santi_pil.paste(frame_pil,(0,0),mask=Image.fromarray(santi_mask))
        #opencvで表示するとき用にRGB➡BGR
        santi_pil = Image.fromarray(cv2.cvtColor(np.array(santi_pil), cv2.COLOR_RGB2BGR))

        #二値作成（santiを2値化）
        niti = np.where(np.array(santi_pil.convert("L"))<255,0,255)
        niti_pil = Image.fromarray(niti.astype(np.uint8))

        return niti_pil, santi_pil, average_color

    def calc_average_color(self,white,rr,cc):
        #whiteをグレースケール化し，thresholdを基準に料理領域とそれ以外を2値化
        not_white = white[rr,cc]
        B_ave = np.mean(not_white[:,0])
        G_ave = np.mean(not_white[:,1])
        R_ave = np.mean(not_white[:,2])
        return np.array([R_ave,G_ave,B_ave])

    #all_mask_dictにマスクを追加する
    def make_all_dict(self):
        print("皿検出開始")
        mask_count = 0
        rect_mask_list = self.detection_rect()
        for mask in rect_mask_list:
            niti,santi,average_color = self.make_santi_and_niti(mask=mask)
            self.all_mask_dict[mask_count] = {}
            self.all_mask_dict[mask_count]["皿"] = Image.fromarray(mask)
            self.all_mask_dict[mask_count]["二値"] = niti
            self.all_mask_dict[mask_count]["三値"] = santi
            self.all_mask_dict[mask_count]["平均色"] = average_color
            mask_count+=1

        circle_mask_list = self.detection_circle()
        for mask in circle_mask_list:
            niti,santi,average_color = self.make_santi_and_niti(mask=mask)
            self.all_mask_dict[mask_count] = {}
            self.all_mask_dict[mask_count]["皿"] = Image.fromarray(mask)
            self.all_mask_dict[mask_count]["二値"] = niti
            self.all_mask_dict[mask_count]["三値"] = santi
            self.all_mask_dict[mask_count]["平均色"] = average_color
            mask_count+=1
        
        print("皿検出終了")
        return self.all_mask_dict
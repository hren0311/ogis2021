import cv2
import math
import numpy as np
from PIL import Image
import time

from detest_sara import Make_mask
from make_gif import Make_gif

####メモ####

# 円検出がうまくいってないときはdetect_sara.pyの
# min_radiusとmax_radiusをいじると改善するかも
# ➡max_radiusを上げると結構皿見地に処理時間がかかる

#皿検出＆gif合成を行っている時間は自分のデスクトップウィンドウを表示する設計
# ➡自分のPCのデスクトップ背景を真っ黒な画像にしておくと良い

#実行を中止するときは表示されているウィンドウ(testかcamera)の上でqボタンを押す
###########


#カメラ取得用
cap = cv2.VideoCapture(0)

#カメラのframeサイズを自分のPCのウィンドウサイズに拡大する
#実験時は1920*1200
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920) # カメラ画像の横幅を1280に設定
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1200) # カメラ画像の縦幅を720に設定
print("カメラ取得完了")


###設定用パラメタータ###
start_time = time.time()
elapsed_time_last = 0
elapsed_time_last_2 = 0
#gifを作成する時間の間隔
get_interval_gif = 20
#maskを作成する時間の間隔
get_interval_mask = 20

#マスクを作成しはじめるまでbeep音を鳴らす用
get_interval_beep = 16
#####################

####
cv2.namedWindow('test', cv2.WINDOW_NORMAL)
cv2.setWindowProperty("test",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
####

print("初期読み込み開始")
#最初
all_black = np.zeros((1200,1920),dtype=np.uint8)
#gif読み込み
make_gif = Make_gif()
while True:
    ret, frame = cap.read()
    cv2.imshow("camera",frame)
    make_mask = Make_mask(frame=frame)
    json_np_new = make_mask.make_all_dict()
    if json_np_new != 0:
        json_np = json_np_new
        im_list,fps = make_gif.make_gif_image(json_np=json_np)
        break

print("初期読み込み終了")

num = 0
while True:

    ret, frame = cap.read()
    cv2.imshow("camera",frame)
    elapsed_time = int(time.time() - start_time)
    
    #mask取得前に（get_intarval_mask-get_interval_beep）秒，beep音を鳴らす
    # if elapsed_time - elapsed_time_last_2 >= get_interval_beep:
    #     winsound.Beep(1000, 10)
    
    if elapsed_time - elapsed_time_last >= get_interval_gif:
        print("{} seconds passed".format(elapsed_time))
        cv2.destroyAllWindows()
        time.sleep(0.1)
        for i in range(50):
            ret, frame = cap.read()
            
        ###
        make_mask = Make_mask(frame=frame)
        json_np_new = make_mask.make_all_dict()
        if json_np_new != 0:
            json_np = json_np_new
            
        im_list,fps = make_gif.make_gif_image(json_np)
        #初期化
        num = 0
        elapsed_time_last = elapsed_time
        elapsed_time_last_2 = elapsed_time

    #初期化
    if num >= len(im_list):
        num = 0
    
    img = cv2.cvtColor(im_list[num], cv2.COLOR_RGB2BGR)
    cv2.namedWindow('test', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("test",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    cv2.imshow("test",img)

    #実際に表示している画像を保存
    # cv2.imwrite("imgs/{}giff.jpg".format(num),img)

    num+=1
    time.sleep(1/fps)

    #Qキーで終了
    key = cv2.waitKey(10)
    if key == ord("q"):
        break

cv2.destroyAllWindows()
import cv2
import numpy as np

def close_wins():
    cv2.waitKey()
    cv2.destroyAllWindows()

def morphology_close(img, k1, k2, iter_):
    k = np.ones((k1,k2), np.uint8)
    result = cv2.morphologyEx(img, cv2.MORPH_CLOSE, k, iterations=iter_)
    #cv2.imshow("原圖", img)
    #cv2.imshow("結果圖-1", result)
    return result

def get_face_l_indices():
    return (95, 250), (330, 460)

def get_face_r_indices():
    return (170, 318), (550, 680)

def retrieve_clear_faces(img):
    # 從原圖擷出照片中兩位女性的臉部 ROI (整體做閉運算後，兩位女性臉部都失真太多)
    #face_l = img[95:250, 330:460] # h, w
    h, w = get_face_l_indices()
    face_l = img[h[0]:h[1], w[0]:w[1]] # h, w
    #cv2.imshow("face - left", face_l) # 顯示左邊女性人臉
    
    face_r = img[170:318, 550:680] # h, w
    h, w = get_face_r_indices()
    face_r = img[h[0]:h[1], w[0]:w[1]]
    #cv2.imshow("face - right", face_r) # 顯示右邊女性人臉
    
    # (suggested) face_l: (k1,k2,iter_) => (2, 2, 5)
    # 以上述參數對 左邊女性人臉 做閉運算
    # 可得到 較不失真 且 同時去除線條 的均衡
    '''
    # test
    for i in range(3,7):
        morphology_close(face_l, 2, 2, i)
        close_wins()
    '''
    clear_face_l = morphology_close(face_l, 2, 2, 5)
    #close_wins()
    clear_face_r = morphology_close(face_r, 2, 2, 5)
    #close_wins()
    return clear_face_l, clear_face_r

def paste_face_on_bg(clear_face, background_image, position):
    choices = ("left", "right")
    if position in choices:
        if position == choices[0]:
            h, w = get_face_l_indices()
        elif position == choices[1]:
            h, w = get_face_r_indices()
        background_image[h[0]:h[1], w[0]:w[1]] = clear_face
        return background_image
    else:
        print("Invalid parameter: 'position' should be right or left.")
        return None
    
if __name__ == "__main__":
    img_path = "../res/People/example_01.jpg"
    img = cv2.imread(img_path)
    
    # << solution 1 >>
    # 閉運算: 先保留較多臉部原始訊息，再試法去掉衣服上的線條--(這部分沒做)
    close_op_result = morphology_close(img, 3, 3, 3) # 太多線條, 臉部較不失真
    cv2.imshow("結果圖 (solution 1)", close_op_result)
    close_wins()

    #-----------------------------------------------------------------------
    # << solution 2 >>
    # 閉運算: 先把整體所有線條去掉，再試法對臉 denoise，再貼回去
    close_op_result = morphology_close(img, 4, 4, 3) # 線條幾乎去除, 臉部失真偏嚴重
    #cv2.imshow("處理線條後的整體圖 (solution 2)", close_op_result)
    #close_wins()
    
    clear_face_l, clear_face_r = retrieve_clear_faces(img) 
    #cv2.imshow("較清晰、去除雜線的左方人臉 (solution 2)", clear_face_l)
    #close_wins()
    #cv2.imshow("較清晰、去除雜線的右方人臉 (solution 2)", clear_face_r)
    #close_wins()
    
    pasted_result = paste_face_on_bg(clear_face_l, close_op_result, "left")
    pasted_result = paste_face_on_bg(clear_face_r, pasted_result, "right")
    cv2.imshow("結果圖 (solution 2)", pasted_result)
    close_wins()
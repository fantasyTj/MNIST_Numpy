import numpy as np
from PIL import Image
def split(img):
    flatten_img=np.sum(img,axis=0)
    flatten_img=np.asarray(flatten_img>40,dtype=np.uint8)
    # 分别寻找左右边界
    r_moved_flatten_img=np.zeros_like(flatten_img)
    r_moved_flatten_img[1:]=flatten_img[:-1]
    l_edge=((r_moved_flatten_img-flatten_img)==255)

    l_moved_flatten_img = np.zeros_like(flatten_img)
    l_moved_flatten_img[:-1] = flatten_img[1:]
    r_edge = ((l_moved_flatten_img - flatten_img) == 255)
    edge=l_edge+r_edge

    number_num=np.sum(edge)//2
    edge_idx =[]
    for i in range(flatten_img.shape[0]):
        if edge[i]!=0:
            edge_idx.append(i)
    pro_img=np.zeros((number_num,28,28))

    # print(edge_idx)

    for i in range(number_num):
        s_img=img[:,edge_idx[2*i]:edge_idx[2*i+1]]

        flatten_img=np.sum(s_img,axis=1)
        o_edge,b_edge=0,flatten_img.shape[0]-1
        for j in range(1,flatten_img.shape[0]):
            if flatten_img[j]>0 and flatten_img[j-1]==0:
                o_edge=j
            if flatten_img[j]==0 and flatten_img[j-1]>0:
                b_edge=j

        width=edge_idx[2*i+1]-edge_idx[2*i]
        height=b_edge-o_edge
        if height>width:
            resize_size=(int(24*(width/height)),24)
        else:
            resize_size=(24,int(24*(height/width)))

        pil_img = Image.fromarray(np.uint8(s_img[o_edge:b_edge,:]))
        pil_img = pil_img.resize(resize_size)

        img_array=np.array(pil_img.convert('L'))

        pro_img[i,(14-img_array.shape[0]//2):(14-img_array.shape[0]//2)+img_array.shape[0],(14-img_array.shape[1]//2):(14-img_array.shape[1]//2)+img_array.shape[1]]=img_array

        # pil_img_=Image.fromarray(np.uint8(pro_img[i]))
        # pil_img_.show()

    return pro_img

if __name__=='__main__':
    a=np.array([[0,111,2,0,3,0],[0,1,3,0,2,0]])
    b=split(a)
    print(b.shape)


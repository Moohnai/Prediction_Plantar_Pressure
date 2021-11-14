import numpy as np

def DataPreprocess(data):
    H_L , H_R = data
    i=0
    j=0
    L=[]
    data_processed=[]
    label=[]
    t=5

    for i in range(len(H_L)) :
        for j in range(len(H_L[i])):
            a=H_L[i][j].shape[2]
            b=H_R[i][j].shape[2]
            L.append(a)
            L.append(b)
            d1=H_L[i][j].shape[0]
            d2=H_L[i][j].shape[1]
            iM =np.zeros(shape=(47 , 32 , H_L[i][j].shape[2]))
            iM[0:d1 , 0:d2 , : ]=(H_L[i][j])
            H_L[i][j] = iM
            # plt.figure()
            # plt.imshow(H_L[i][j][:, :, 200])
            # plt.show()
            # print(H_L[i][j][20, 10 , 100])
            d1 = H_R[i][j].shape[0]
            d2 = H_R[i][j].shape[1]
            iM = np.zeros(shape=(47, 32 , H_R[i][j].shape[2]))
            iM[0:d1, 0:d2, :] = H_R[i][j]
            H_R[i][j] = iM



            z=0
            H_L[i][j] = H_L[i][j][:,:,::5]
            for l in range ((H_L[i][j].shape[2])//t):
                if (l+1)*t<H_L[i][j].shape[2]:
                    data_processed.append(H_L[i][j][:, :, z:(l + 1) * t])
                    a=H_L[i][j][:,:,(l+1)*t]
                    label.append(a)
                    z=(l+1)*t

            z = 0
            H_R[i][j] = H_R[i][j][:, :, ::5]
            for l in range((H_R[i][j].shape[2]) // t):
                if (l + 1) * t < H_R[i][j].shape[2]:
                    data_processed.append(H_R[i][j][:, :, z:(l + 1) * t])
                    a = H_R[i][j][:,:,(l+1)*t]
                    label.append(a)
                    z = (l+1)*t

    return data_processed, label
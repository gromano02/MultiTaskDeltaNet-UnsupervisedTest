import numpy as np # linear algebra
import pandas as pd
import os

from PIL import Image
import re

from datasets.data_phy import REGDataset

def load_dataset():

    def normalize(input_data):
        return (input_data.astype(np.float32))/255.0

    def denormalize(input_data):
        input_data = (input_data) * 255
        return input_data.astype(np.uint8)

    palette = np.array([[1, 0, 0],  # "no change"
                    [0, 1, 0],  # "over extrude(1)"
                    [0, 0, 1]], dtype='float32')  # "under extrude (-1)"

    palette2 = np.array([[1, 1, 1],  # "no change"
                    [0.5, 0, 0],  # "over extrude(1)"
                    [0, 0.5, 0]], dtype='float32')  # "under extrude (-1)"

    palette3 = np.array([[1, 1, 1],  # "no change"
                    [0.5, 0, 0],  # "appearing(1)"
                    [0, 0.5, 0],  # "disappearing(-1)"
                    [0.4,0.4,0.5]], dtype='float32')  # "overlap(2)"  [0.4,0.4,0.5]


    path='/home/yun13001/dataset/Carbon/tianyu_new_data/New_distribution/'

    dirnames = os.listdir(path)
    dirnames.remove('kinetic_curves.xlsx')
    dirnames=sorted(dirnames, key=lambda s: float(re.findall(r'\d+', s)[0]))
    print(dirnames)


    file_names={}
    for dirname in dirnames:
        for root, dirnames, filenames_x in os.walk(path+'/'+dirname+'/results/img'):
            break
        filenames_x = sorted(filenames_x, key=lambda s: float(re.findall(r'\d+', s)[2]))
        if dirname == '201' or dirname == '203':
            filenames_x.pop(0)
        else:
            filenames_x.pop(0)
            filenames_x.pop(0)

        file_names[dirname]=filenames_x


    #for key, value in file_names.items():
    #     print(key, value)

    train_name=['102_R1','102_R2', '302']
    #val_name = ['103','301']
    val_name1 = ['103']
    val_name2 = ['301']
    test_name= ['201','203']

    ##### Len 
    data_len={}
    data_len['101']=65.29
    data_len['103']=152.81
    data_len['201']=105.51
    data_len['203']=95.89
    data_len['301']=148.836
    data_len['302']=226.586
    data_len['102_R1']=227.32
    data_len['102_R2']=202.06    

    ###### y value, unnorm Avrami value
    data_y={}
    data_max={}
    data_min={}
   
    ### 101
    df=pd.read_excel(path+'/kinetic_curves.xlsx',sheet_name='V101', usecols='C,H')
    value=df.to_numpy()
    value_T =(value[3:19+3,0])
    value_y =(value[3:19+3,1])
    T_max = max(value_T)
    t= value_T/T_max
    y_max = max(value_y)
    y_min = min(value_y)
    k=9.22
    n=1.98
    name ='101'
    data_max[name]=y_max
    data_min[name]=y_min

    for i in range(len(value_y)):
        #nn= 1-np.exp(-(k*(t[i]**n)))*(y_max-y_min)+y_min
        nn= 1-np.exp(-(k*(t[i]**n)))
        name='101'
        if name not in data_y:
            data_y[name]=[nn]
        else:
            data_y[name].append(nn)


    ### 103
    df=pd.read_excel(path+'/kinetic_curves.xlsx',sheet_name='V103', usecols='C,H')
    value=df.to_numpy()
    value_T =(value[3:46+3,0])
    value_y =(value[3:46+3,1])
    T_max = max(value_T)
    t= value_T/T_max
    y_max = max(value_y)
    y_min = min(value_y)
    k=4.67
    n=1.63
    name='103'
    data_max[name]=y_max
    data_min[name]=y_min

    for i in range(len(value_y)):
        #nn= 1-np.exp(-(k*(t[i]**n)))*(y_max-y_min)+y_min
        nn= 1-np.exp(-(k*(t[i]**n)))
        name='103'
        if name not in data_y:
            data_y[name]=[nn]
        else:
            data_y[name].append(nn)   

    ### 201
    df=pd.read_excel(path+'/kinetic_curves.xlsx',sheet_name='V201', usecols='C,H')
    value=df.to_numpy()
    value_T =(value[2:27+2,0])
    value_y =(value[2:27+2,1])
    T_max = max(value_T)
    t= value_T/T_max
    y_max = max(value_y)
    y_min = min(value_y)
    k=5.56
    n=2.24
    name='201'
    data_max[name]=y_max
    data_min[name]=y_min

    for i in range(len(value_y)):
        #nn= 1-np.exp(-(k*(t[i]**n)))*(y_max-y_min)+y_min
        nn= 1-np.exp(-(k*(t[i]**n)))
        name='201'
        if name not in data_y:
            data_y[name]=[nn]
        else:
            data_y[name].append(nn)

    ### 203
    df=pd.read_excel(path+'/kinetic_curves.xlsx',sheet_name='V203', usecols='C,H')
    value=df.to_numpy()
    value_T =(value[2:14+2,0])
    value_y =(value[2:14+2,1])
    T_max = max(value_T)
    t= value_T/T_max
    y_max = max(value_y)
    y_min = min(value_y)
    k=3.88
    n=2.52
    name='203'
    data_max[name]=y_max
    data_min[name]=y_min

    for i in range(len(value_y)):
        #nn= 1-np.exp(-(k*(t[i]**n)))*(y_max-y_min)+y_min
        nn= 1-np.exp(-(k*(t[i]**n)))
        name='203'
        if name not in data_y:
            data_y[name]=[nn]
        else:
            data_y[name].append(nn)

    ### 301
    df=pd.read_excel(path+'/kinetic_curves.xlsx',sheet_name='V301', usecols='C,H')
    value=df.to_numpy()
    value_T =(value[3:18+3,0])
    value_y =(value[3:18+3,1])
    T_max = max(value_T)
    t= value_T/T_max
    y_max = max(value_y)
    y_min = min(value_y)
    k=9.90
    n=3.36
    name='301'
    data_max[name]=y_max
    data_min[name]=y_min

    for i in range(len(value_y)):
        #nn= 1-np.exp(-(k*(t[i]**n)))*(y_max-y_min)+y_min
        nn= 1-np.exp(-(k*(t[i]**n)))
        name='301'
        if name not in data_y:
            data_y[name]=[nn]
        else:
            data_y[name].append(nn)
            
    ### 302
    df=pd.read_excel(path+'/kinetic_curves.xlsx',sheet_name='V302', usecols='C,H')
    value=df.to_numpy()
    value_T =(value[3:19+3,0])
    value_y =(value[3:19+3,1])
    T_max = max(value_T)
    t= value_T/T_max
    y_max = max(value_y)
    y_min = min(value_y)
    k=6.58
    n=2.62
    name='302'
    data_max[name]=y_max
    data_min[name]=y_min

    for i in range(len(value_y)):
        #nn= 1-np.exp(-(k*(t[i]**n)))*(y_max-y_min)+y_min
        nn= 1-np.exp(-(k*(t[i]**n)))
        name='302'
        if name not in data_y:
            data_y[name]=[nn]
        else:
            data_y[name].append(nn)            

    ### 102_R1
    df=pd.read_excel(path+'/kinetic_curves.xlsx',sheet_name='V102-R1', usecols='C,H')
    value=df.to_numpy()
    value_T =(value[3:52+3,0])
    value_y =(value[3:52+3,1])
    T_max = max(value_T)
    t= value_T/T_max
    y_max = max(value_y)
    y_min = min(value_y)
    k=5.80
    n=2.39
    name='102_R1'
    data_max[name]=y_max
    data_min[name]=y_min

    for i in range(len(value_y)):
        #nn= 1-np.exp(-(k*(t[i]**n)))*(y_max-y_min)+y_min
        nn= 1-np.exp(-(k*(t[i]**n)))
        name='102_R1'
        if name not in data_y:
            data_y[name]=[nn]
        else:
            data_y[name].append(nn)

    ### 102_R2
    df=pd.read_excel(path+'/kinetic_curves.xlsx',sheet_name='V102-R2', usecols='C,H')
    value=df.to_numpy()
    value_T =(value[3:57+3,0])
    value_y =(value[3:57+3,1])
    T_max = max(value_T)
    t= value_T/T_max
    y_max = max(value_y)
    y_min = min(value_y)
    k=5.80
    n=3.16
    name='102_R2'
    data_max[name]=y_max
    data_min[name]=y_min

    for i in range(len(value_y)):
        #nn= 1-np.exp(-(k*(t[i]**n)))*(y_max-y_min)+y_min
        nn= 1-np.exp(-(k*(t[i]**n)))
        name='102_R2'
        if name not in data_y:
            data_y[name]=[nn]
        else:
            data_y[name].append(nn)

    #### train:
    ##### Training dataset:
    train_ref=[]
    train_res=[]

    train_ref_y = []     ### training Avrami value (for ref img)
    train_res_y = []     ### training Avrami value (for res img)    

    train_Max=[]
    train_Min=[]
    train_L =[]

    train_label1=[]  ### training label
    train_label2=[]
    train_map1=[]    ### visulization label
    train_map2=[]

    for name in train_name:    #### Loop all the folders

        ref=[]
        label1=[]
        label2=[]


        imgs = file_names[name]
        for i in range(len(imgs)):
            x_ref = Image.open(path+name+'/results/img/'+imgs[i]).convert('RGB')
            x_label1 = Image.open(path+name+'/results/area1/'+imgs[i]).convert('RGB')
            x_label2 = Image.open(path+name+'/results/area2/'+imgs[i]).convert('RGB')

            ref.append(x_ref)
            label1.append(x_label1)
            label2.append(x_label2)

            x_ref = np.array(x_ref)
            x_label1 = np.array(x_label1)
            x_label2 = np.array(x_label2)


        ref = np.array(ref)
        label1 = np.array(label1)
        label2 = np.array(label2)

        ref = normalize(ref)
        label1=normalize(label1)
        label2=normalize(label2)

        max_ = data_max[name]
        min_ = data_min[name]
        y    = data_y[name]
        l    = data_len[name]

        for m in range(len(ref)):

            numb =m

            #for n in range(numb):
            for n in range(len(ref)):

                train_ref.append(ref[m])
                train_res.append(ref[n])

                ###### Avrami value
                train_ref_y.append(y[m])
                train_res_y.append(y[n])

                ###### L, max and min value
                train_L.append(l)
                train_Max.append(max_)
                train_Min.append(min_)


                ###### Label1 ######
                x_ref = label1[m]
                x_res = label1[n]

                #### overlap
                img = x_ref + x_res
                img_test=img[:,:,0]
                img_ol=img_test.copy()
                img_ol[(img_test==0)]=4.0
                ####

                #### difference
                img = x_ref - x_res
                img_test=img[:,:,0]
                img_tt=img_test.copy()
                img_tt[(img_test<0)]=2
                img_tt[(img_ol==4.0)]=3

                img_tt=np.int32(img_tt)
                mask = palette3[img_tt.ravel()].reshape(img.shape)
                ####

                train_label1.append(img_tt)
                train_map1.append(mask)
                ######  ######


                ###### Label2 ######
                x_ref = label2[m]
                x_res = label2[n]

                #### overlap
                img = x_ref + x_res
                img_test=img[:,:,0]
                img_ol=img_test.copy()
                img_ol[(img_test==0)]=4.0
                ####

                #### difference
                img = x_ref - x_res
                img_test=img[:,:,0]
                img_tt=img_test.copy()
                img_tt[(img_test<0)]=2
                img_tt[(img_ol==4.0)]=3

                img_tt=np.int32(img_tt)
                mask = palette3[img_tt.ravel()].reshape(img.shape)
                ####

                train_label2.append(img_tt)
                train_map2.append(mask)
                ######  ######

    print('train test!')
    train_ref = np.array(train_ref)
    train_res = np.array(train_res)

    train_ref = denormalize(train_ref)
    train_res = denormalize(train_res)

    train_label1 = np.array(train_label1)
    train_label2 = np.array(train_label2)
    train_map1 = np.array(train_map1)
    train_map2 = np.array(train_map2)

    train_ref_y = np.array(train_ref_y)
    train_res_y = np.array(train_res_y)
    train_L    = np.array(train_L)
    train_Max  = np.array(train_Max)
    train_Min  = np.array(train_Min)
    train_pi   = np.repeat(3.1415, len(train_ref))

    print(train_ref.shape)

    #### validation1:
    ##### Validation dataset1:
    val_ref=[]
    val_res=[]

    val_ref_y = []     ### val Avrami value (for ref img)
    val_res_y = []     ### val Avrami value (for res img)    

    val_Max=[]
    val_Min=[]
    val_L =[]

    val_label1=[]  ### validation label
    val_label2=[]
    val_map1=[]    ### visulization label
    val_map2=[]

    for name in val_name1:    #### Loop all the folders

        ref=[]
        label1=[]
        label2=[]

        imgs = file_names[name]
        for i in range(len(imgs)):
            x_ref = Image.open(path+name+'/results/img/'+imgs[i]).convert('RGB')
            x_label1 = Image.open(path+name+'/results/area1/'+imgs[i]).convert('RGB')
            x_label2 = Image.open(path+name+'/results/area2/'+imgs[i]).convert('RGB')

            ref.append(x_ref)
            label1.append(x_label1)
            label2.append(x_label2)

            x_ref = np.array(x_ref)
            x_label1 = np.array(x_label1)
            x_label2 = np.array(x_label2)


        ref = np.array(ref)
        label1 = np.array(label1)
        label2 = np.array(label2)

        ref = normalize(ref)
        label1=normalize(label1)
        label2=normalize(label2)

        max_ = data_max[name]
        min_ = data_min[name]
        y    = data_y[name]
        l    = data_len[name]

        for m in range(len(ref)):

            numb =m

            #for n in range(numb):
            for n in range(len(ref)):

                val_ref.append(ref[m])
                val_res.append(ref[n])

                ###### Avrami value
                val_ref_y.append(y[m])
                val_res_y.append(y[n])

                ###### L, max and min value
                val_L.append(l)
                val_Max.append(max_)
                val_Min.append(min_)

                ###### Label1 ######
                x_ref = label1[m]
                x_res = label1[n]

                #### overlap
                img = x_ref + x_res
                img_test=img[:,:,0]
                img_ol=img_test.copy()
                img_ol[(img_test==0)]=4.0
                ####

                #### difference
                img = x_ref - x_res
                img_test=img[:,:,0]
                img_tt=img_test.copy()
                img_tt[(img_test<0)]=2
                img_tt[(img_ol==4.0)]=3

                img_tt=np.int32(img_tt)
                mask = palette3[img_tt.ravel()].reshape(img.shape)
                ####

                val_label1.append(img_tt)
                val_map1.append(mask)
                ######  ######


                ###### Label2 ######
                x_ref = label2[m]
                x_res = label2[n]

                #### overlap
                img = x_ref + x_res
                img_test=img[:,:,0]
                img_ol=img_test.copy()
                img_ol[(img_test==0)]=4.0
                ####

                #### difference
                img = x_ref - x_res
                img_test=img[:,:,0]
                img_tt=img_test.copy()
                img_tt[(img_test<0)]=2
                img_tt[(img_ol==4.0)]=3

                img_tt=np.int32(img_tt)
                mask = palette3[img_tt.ravel()].reshape(img.shape)
                ####

                val_label2.append(img_tt)
                val_map2.append(mask)
                ######  ######

    print('validation test1!')
    val_ref = np.array(val_ref)
    val_res = np.array(val_res)

    val_ref = denormalize(val_ref)
    val_res = denormalize(val_res)

    val_label1 = np.array(val_label1)
    val_label2 = np.array(val_label2)
    val_map1 = np.array(val_map1)
    val_map2 = np.array(val_map2)

    val_Max =  np.array(val_Max)
    val_Min =  np.array(val_Min)
    val_L =  np.array(val_L)
    val_ref_y = np.array(val_ref_y)
    val_res_y = np.array(val_res_y)
    val_pi        = np.repeat(3.1415, len(val_ref))

    print(val_ref.shape)
    val_dataset1 = REGDataset(val_ref,val_res,val_label1,val_label2,val_ref_y,val_res_y,val_Max,val_Min,val_L,val_pi,img_size=256,is_train=False,to_tensor=True)

    
    #### validation2:
    ##### Validation dataset2:
    val_ref=[]
    val_res=[]

    val_ref_y = []     ### val Avrami value (for ref img)
    val_res_y = []     ### val Avrami value (for res img)    

    val_Max=[]
    val_Min=[]
    val_L =[]

    val_label1=[]  ### validation label
    val_label2=[]
    val_map1=[]    ### visulization label
    val_map2=[]

    for name in val_name2:    #### Loop all the folders

        ref=[]
        label1=[]
        label2=[]

        imgs = file_names[name]
        for i in range(len(imgs)):
            x_ref = Image.open(path+name+'/results/img/'+imgs[i]).convert('RGB')
            x_label1 = Image.open(path+name+'/results/area1/'+imgs[i]).convert('RGB')
            x_label2 = Image.open(path+name+'/results/area2/'+imgs[i]).convert('RGB')

            ref.append(x_ref)
            label1.append(x_label1)
            label2.append(x_label2)

            x_ref = np.array(x_ref)
            x_label1 = np.array(x_label1)
            x_label2 = np.array(x_label2)


        ref = np.array(ref)
        label1 = np.array(label1)
        label2 = np.array(label2)

        ref = normalize(ref)
        label1=normalize(label1)
        label2=normalize(label2)

        max_ = data_max[name]
        min_ = data_min[name]
        y    = data_y[name]
        l    = data_len[name]


        for m in range(len(ref)):

            numb =m

            #for n in range(numb):
            for n in range(len(ref)):

                val_ref.append(ref[m])
                val_res.append(ref[n])

                ###### Avrami value
                val_ref_y.append(y[m])
                val_res_y.append(y[n])

                ###### L, max and min value
                val_L.append(l)
                val_Max.append(max_)
                val_Min.append(min_)

                ###### Label1 ######
                x_ref = label1[m]
                x_res = label1[n]

                #### overlap
                img = x_ref + x_res
                img_test=img[:,:,0]
                img_ol=img_test.copy()
                img_ol[(img_test==0)]=4.0
                ####

                #### difference
                img = x_ref - x_res
                img_test=img[:,:,0]
                img_tt=img_test.copy()
                img_tt[(img_test<0)]=2
                img_tt[(img_ol==4.0)]=3

                img_tt=np.int32(img_tt)
                mask = palette3[img_tt.ravel()].reshape(img.shape)
                ####

                val_label1.append(img_tt)
                val_map1.append(mask)
                ######  ######


                ###### Label2 ######
                x_ref = label2[m]
                x_res = label2[n]

                #### overlap
                img = x_ref + x_res
                img_test=img[:,:,0]
                img_ol=img_test.copy()
                img_ol[(img_test==0)]=4.0
                ####

                #### difference
                img = x_ref - x_res
                img_test=img[:,:,0]
                img_tt=img_test.copy()
                img_tt[(img_test<0)]=2
                img_tt[(img_ol==4.0)]=3

                img_tt=np.int32(img_tt)
                mask = palette3[img_tt.ravel()].reshape(img.shape)
                ####

                val_label2.append(img_tt)
                val_map2.append(mask)
                ######  ######

    print('validation test2!')
    val_ref = np.array(val_ref)
    val_res = np.array(val_res)

    val_ref = denormalize(val_ref)
    val_res = denormalize(val_res)

    val_label1 = np.array(val_label1)
    val_label2 = np.array(val_label2)
    val_map1 = np.array(val_map1)
    val_map2 = np.array(val_map2)

    val_Max =  np.array(val_Max)
    val_Min =  np.array(val_Min)
    val_L =  np.array(val_L)
    val_ref_y = np.array(val_ref_y)
    val_res_y = np.array(val_res_y)
    val_pi        = np.repeat(3.1415, len(val_ref))

    print(val_ref.shape)
    val_dataset2 = REGDataset(val_ref,val_res,val_label1,val_label2,val_ref_y,val_res_y,val_Max,val_Min,val_L,val_pi,img_size=256,is_train=False,to_tensor=True)

    #### test:
    ##### Test dataset:
    test_ref=[]
    test_res=[]

    test_ref_y = []     ### val Avrami value (for ref img)
    test_res_y = []     ### val Avrami value (for res img)

    test_Max=[]
    test_Min=[]
    test_L =[]

    test_label1=[]  ### validation label
    test_label2=[]
    test_map1=[]    ### visulization label
    test_map2=[]

    for name in test_name:    #### Loop all the folders

        ref=[]
        label1=[]
        label2=[]

        imgs = file_names[name]
        for i in range(len(imgs)):
            x_ref = Image.open(path+name+'/results/img/'+imgs[i]).convert('RGB')
            x_label1 = Image.open(path+name+'/results/area1/'+imgs[i]).convert('RGB')
            x_label2 = Image.open(path+name+'/results/area2/'+imgs[i]).convert('RGB')

            ref.append(x_ref)
            label1.append(x_label1)
            label2.append(x_label2)

            x_ref = np.array(x_ref)
            x_label1 = np.array(x_label1)
            x_label2 = np.array(x_label2)


        ref = np.array(ref)
        label1 = np.array(label1)
        label2 = np.array(label2)

        ref = normalize(ref)
        label1=normalize(label1)
        label2=normalize(label2)

        max_ = data_max[name]
        min_ = data_min[name]
        y    = data_y[name]
        l    = data_len[name]


        for m in range(len(ref)):

            numb =m

            #for n in range(numb):
            for n in range(len(ref)):

                test_ref.append(ref[m])
                test_res.append(ref[n])

                ###### Label1 ######
                x_ref = label1[m]
                x_res = label1[n]

                ###### Avrami value
                test_ref_y.append(y[m])
                test_res_y.append(y[n])

                ###### L, max and min value
                test_L.append(l)
                test_Max.append(max_)
                test_Min.append(min_)

                #### overlap
                img = x_ref + x_res
                img_test=img[:,:,0]
                img_ol=img_test.copy()
                img_ol[(img_test==0)]=4.0
                ####

                #### difference
                img = x_ref - x_res
                img_test=img[:,:,0]
                img_tt=img_test.copy()
                img_tt[(img_test<0)]=2
                img_tt[(img_ol==4.0)]=3

                img_tt=np.int32(img_tt)
                mask = palette3[img_tt.ravel()].reshape(img.shape)
                ####

                test_label1.append(img_tt)
                test_map1.append(mask)
                ######  ######


                ###### Label2 ######
                x_ref = label2[m]
                x_res = label2[n]

                #### overlap
                img = x_ref + x_res
                img_test=img[:,:,0]
                img_ol=img_test.copy()
                img_ol[(img_test==0)]=4.0
                ####

                #### difference
                img = x_ref - x_res
                img_test=img[:,:,0]
                img_tt=img_test.copy()
                img_tt[(img_test<0)]=2
                img_tt[(img_ol==4.0)]=3

                img_tt=np.int32(img_tt)
                mask = palette3[img_tt.ravel()].reshape(img.shape)
                ####

                test_label2.append(img_tt)
                test_map2.append(mask)
                ######  ######

        
    print('testing test!')
    test_ref = np.array(test_ref)
    test_res = np.array(test_res)

    test_ref = denormalize(test_ref)
    test_res = denormalize(test_res)

    test_label1 = np.array(test_label1)
    test_label2 = np.array(test_label2)
    test_map1 = np.array(test_map1)
    test_map2 = np.array(test_map2)

    test_ref_y = np.array(test_ref_y)
    test_res_y = np.array(test_res_y)
    test_L    = np.array(test_L)
    test_Max  = np.array(test_Max)
    test_Min  = np.array(test_Min)

    test_pi   = np.repeat(3.1415, len(test_ref))

    print(test_ref.shape)

    train_dataset = REGDataset(train_ref,train_res,train_label1,train_label2,train_ref_y,train_res_y,train_Max,train_Min,train_L,train_pi,img_size=256,is_train=True,to_tensor=True)
    test_dataset= REGDataset(test_ref,test_res,test_label1,test_label2,test_ref_y,test_res_y,test_Max,test_Min,test_L,test_pi,img_size=256,is_train=False,to_tensor=True)


    return train_dataset, val_dataset1, val_dataset2, test_dataset

def get_neck(option, in_channel_list): #4
    if option['neck'].lower() == 'aspp': #4
        from model.neck.neck import aspp_neck 
        neck = aspp_neck(in_channel_list, option['neck_channel'])
    elif option['neck'].lower() == 'basic': #4
        from model.neck.neck import basic_neck  #4
        neck = basic_neck(in_channel_list, option['neck_channel'])  #5

    return neck #6

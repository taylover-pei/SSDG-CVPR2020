import torch

def Real_AdLoss(discriminator_out, criterion, shape_list):
    # generate ad_label
    ad_label1_index = torch.LongTensor(shape_list[0], 1).fill_(0)
    ad_label1 = ad_label1_index.cuda()
    ad_label2_index = torch.LongTensor(shape_list[1], 1).fill_(1)
    ad_label2 = ad_label2_index.cuda()
    ad_label3_index = torch.LongTensor(shape_list[2], 1).fill_(2)
    ad_label3 = ad_label3_index.cuda()
    ad_label = torch.cat([ad_label1, ad_label2, ad_label3], dim=0).view(-1)

    real_adloss = criterion(discriminator_out, ad_label)
    return real_adloss

def Fake_AdLoss(discriminator_out, criterion, shape_list):
    # generate ad_label
    ad_label1_index = torch.LongTensor(shape_list[0], 1).fill_(0)
    ad_label1 = ad_label1_index.cuda()
    ad_label2_index = torch.LongTensor(shape_list[1], 1).fill_(1)
    ad_label2 = ad_label2_index.cuda()
    ad_label3_index = torch.LongTensor(shape_list[2], 1).fill_(2)
    ad_label3 = ad_label3_index.cuda()
    ad_label = torch.cat([ad_label1, ad_label2, ad_label3], dim=0).view(-1)

    fake_adloss = criterion(discriminator_out, ad_label)
    return fake_adloss

def AdLoss_Limited(discriminator_out, criterion, shape_list):
    # generate ad_label
    ad_label2_index = torch.LongTensor(shape_list[0], 1).fill_(0)
    ad_label2 = ad_label2_index.cuda()
    ad_label3_index = torch.LongTensor(shape_list[1], 1).fill_(1)
    ad_label3 = ad_label3_index.cuda()
    ad_label = torch.cat([ad_label2, ad_label3], dim=0).view(-1)

    real_adloss = criterion(discriminator_out, ad_label)
    return real_adloss

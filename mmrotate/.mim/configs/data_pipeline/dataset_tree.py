### TODO: attention!!! data_root should be ended with "/"!
UCAS_AODv10=dict(
    version='r64_s512',
    data_name='UCAS_AODv10Dataset',
    data_root='/home/zzc/data/ObjDet/UCAS_AOD/processed/v3_cropped/',
    train_image_rdir='train/images/',
    train_label_rdir='train/txt/',
    test_image_rdir='test/images/',
    test_label_rdir='test/txt/',
    data_pipeline='obb_{}_512x512_pipeline',
    num_classes=2
)

DOTAv15=dict(
    version='v1.5',
    data_name='DOTAv1_5Dataset',
    data_root='/home/zzc/data/ObjDet/DOTA/processed/v1_5/',
    train_image_rdir='train/images/',
    train_label_rdir='train/txt/',
    test_image_rdir='val/images/',
    test_label_rdir='val/txt/',
    data_pipeline='obb_{}_1024x1024_pipeline',
    num_classes=16
)

DOTAv15_RT02=dict(
    version='v1.5 random translation 0.2',
    data_name='DOTAv1_5Dataset_RT02',
    data_root='/home/zzc/data/ObjDet/DOTA/processed/v1_5/',
    train_image_rdir='train/images/',
    train_label_rdir='train/txt/',
    test_image_rdir='val/images/',
    test_label_rdir='val/txt/',
    data_pipeline='obb_{}_1024x1024_pipeline',
    num_classes=16
)

DOTAv15_RT04=dict(
    version='v1.5 random translation 0.4',
    data_name='DOTAv1_5Dataset_RT04',
    data_root='/home/zzc/data/ObjDet/DOTA/processed/v1_5/',
    train_image_rdir='train/images/',
    train_label_rdir='train/txt/',
    test_image_rdir='val/images/',
    test_label_rdir='val/txt/',
    data_pipeline='obb_{}_1024x1024_pipeline',
    num_classes=16
)

DOTAv15_RT06=dict(
    version='v1.5 random translation 0.6',
    data_name='DOTAv1_5Dataset_RT06',
    data_root='/home/zzc/data/ObjDet/DOTA/processed/v1_5/',
    train_image_rdir='train/images/',
    train_label_rdir='train/txt/',
    test_image_rdir='val/images/',
    test_label_rdir='val/txt/',
    data_pipeline='obb_{}_1024x1024_pipeline',
    num_classes=16
)

DOTAv15_ms=dict(
    version='v1.5 multi-scale',
    data_name='DOTAv1_5Dataset',
    data_root='/home/zzc/data/ObjDet/DOTA/processed/v1_5_ms/',
    train_image_rdir='train/images/',
    train_label_rdir='train/txt/',
    test_image_rdir='val/images/',
    test_label_rdir='val/txt/',
    data_pipeline='obb_{}_1024x1024_pipeline',
    num_classes=16
)

DOTAv15_HM=dict(
    version='v1.5',
    data_name='DOTA_HM_Dataset',
    data_root='/home/zzc/data/ObjDet/DOTA/processed/v1_5_patch256_gap128/',
    train_image_rdir='train/images/',
    train_label_rdir='train/heatmaps_1.0/',
    test_image_rdir='val/images/',
    test_label_rdir='val/heatmaps_1.0/',
    data_pipeline='obb_{}_256x256_pipeline',
    num_classes=1
)

# DOTAv15_raw=dict(
#     version='v1.5 huge image',
#     data_name='DOTAv1_5Dataset',
#     data_root='/home/zzc/data/ObjDet/DOTA/raw/',
#     train_image_rdir='train/images/',
#     train_label_rdir='train/lableTxt-v1.5/',
#     test_image_rdir='val/images/',
#     test_label_rdir='val/lableTxt-v1.5/',
#     data_pipeline='obb_{}_1024x1024_pipeline',
#     num_classes=16
# )

DOTAv10=dict(
    version='v1.0',
    data_name='DOTADataset',
    data_root='/home/zzc/data/ObjDet/DOTA/processed/v1_0/',
    train_image_rdir='train/images/',
    train_label_rdir='train/txt/',
    test_image_rdir='val/images/',
    test_label_rdir='val/txt/',
    data_pipeline='obb_{}_1024x1024_pipeline',
    num_classes=15
)

DOTAv10_TV=dict(
    version='v1.0-trainval',
    data_name='DOTADataset',
    data_root='/home/zzc/data/ObjDet/DOTA/processed/v1_0/',
    train_image_rdir='trainval/images/',
    train_label_rdir='trainval/txt/',
    test_image_rdir='val/images/',
    test_label_rdir=None,
    data_pipeline='obb_{}_1024x1024_pipeline',
    num_classes=15
)

DOTAv10_ms=dict(
    version='v1.0 multi-scale',
    data_name='DOTADataset',
    data_root='/home/zzc/data/ObjDet/DOTA/processed/v1_0_ms/',
    train_image_rdir='train/images/',
    train_label_rdir='train/txt/',
    test_image_rdir='val/images/',
    test_label_rdir='val/txt/',
    data_pipeline='obb_{}_1024x1024_pipeline',
    num_classes=15
)

DOTAv10_HM=dict(
    version='v1.0',
    data_name='DOTA_HM_Dataset',
    data_root='/home/zzc/data/ObjDet/DOTA/processed/v1_0_patch256_gap128/',
    train_image_rdir='train/images/',
    train_label_rdir='train/heatmaps_1.0/',
    test_image_rdir='val/images/',
    test_label_rdir='val/heatmaps_1.0/',
    data_pipeline='obb_{}_256x256_pipeline',
    num_classes=1
)



# HM_v10=dict(
#     version='v1.0',
#     data_name='DOTA_HM_Dataset',
#     data_root='/home/zzc/data/ObjDet/HM_v1/v1_0_patch256_gap128/',
#     train_image_rdir='trainval/images/',
#     train_label_rdir='trainval/heatmaps_1.0/',
#     test_image_rdir='trainval/images/',
#     test_label_rdir='trainval/heatmaps_1.0/',
#     data_pipeline='hm_{}_256x256_pipeline',
#     num_classes=1
# )
HM_v10=dict(
    version='v1.0',
    data_name='DOTA_HM_Dataset',
    data_root='/home/zzc/data/ObjDet/HM_v1/v2_0_patch256_gap128_FDD/mini4lab_256/',
    train_image_rdir='images/',
    train_label_rdir='heatmaps_1.0/',
    test_image_rdir='images/',
    test_label_rdir='heatmaps_1.0/',
    data_pipeline='hm_{}_256x256_pipeline',
    num_classes=1
)

FDD_v10=dict(
    version='v1.0',
    data_name='FDDv1_0Dataset',
    data_root='/home/zzc/data/ObjDet/HM_v1/v1_0_patch256_gap128/',
    train_image_rdir='trainval/images/',
    train_label_rdir='trainval/txt/',
    test_image_rdir='trainval/images/',
    test_label_rdir='trainval/txt/',
    data_pipeline='obb_{}_256x256_pipeline',
    num_classes=1
)

FDD_v20=dict(
    version='v2.0',
    data_name='FDDv1_0Dataset',
    data_root='/home/zzc/data/ObjDet/HM_v1/v2_0_patch256_gap128_FDD/',
    train_image_rdir='trainval/images/',
    train_label_rdir='trainval/txt/',
    test_image_rdir='trainval/images/',
    test_label_rdir='trainval/txt/',
    data_pipeline='obb_{}_256x256_pipeline',
    num_classes=1
)

FDD_v30=dict(
    version='v3.0',
    data_name='FDDv1_0Dataset',
    data_root='/home/zzc/data/ObjDet/HM_v1/v3_0_patch256_gap128_FDD/',
    train_image_rdir='trainval/images/',
    train_label_rdir='trainval/txt/',
    test_image_rdir='trainval/images/',
    test_label_rdir='trainval/txt/',
    data_pipeline='obb_{}_256x256_pipeline',
    num_classes=1
)

FDD_v40=dict(
    version='v4.0',
    data_name='FDDv1_0Dataset',
    # data_root='/home/zzc/data/ObjDet/HM_v1/v4_0_patch256_balanced_FDD/',
    data_root='/data1/zzc/data/ObjDet/HM_v1/v4_0_patch256_balanced_FDD/',
    train_image_rdir='trainval/images/',
    train_label_rdir='trainval/txt/',
    test_image_rdir='trainval/images/',
    test_label_rdir='trainval/txt/',
    data_pipeline='obb_{}_256x256_pipeline',
    num_classes=1
)

FDD_v20_mini256=dict(
    version='v2.0',
    data_name='FDDv1_0Dataset',
    data_root='/home/zzc/data/ObjDet/HM_v1/v2_0_patch256_gap128_FDD/',
    train_image_rdir='mini4lab_256/images/',
    train_label_rdir='mini4lab_256/txt/',
    test_image_rdir='mini4lab_256/images/',
    test_label_rdir='mini4lab_256/txt/',
    data_pipeline='obb_{}_256x256_pipeline',
    num_classes=1
)

FAIR1Mv2=dict(
    version='v2',
    data_name='FAIR1MDataset',
    data_root='/home/zzc/data/ObjDet/FAIR1M2.0/split/split_ss_fair1m/',
    train_image_rdir='train/images/',
    train_label_rdir='train/annfiles/',
    test_image_rdir='val/images/',
    test_label_rdir='val/annfiles/',
    data_pipeline='obb_{}_1024x1024_pipeline',
    num_classes=37
)

FAIR1Mv2_mini=dict(
    version='v2 mini',
    data_name='FAIR1MDataset',
    data_root='/home/zzc/data/ObjDet/FAIR1M2.0/split/split_ss_fair1m/',
    train_image_rdir='mini4test/images/',
    train_label_rdir='mini4test/annfiles/',
    test_image_rdir='mini4test/images/',
    test_label_rdir='mini4test/annfiles/',
    data_pipeline='obb_{}_1024x1024_pipeline',
    num_classes=37
)

DIORR=dict(
    version='OBB',
    data_name='DIOR_txt_Dataset',
    data_root='/home/zzc/data/ObjDet/DIOR/processed/',
    train_image_rdir='trainval_OBB/images/',
    train_label_rdir='trainval_OBB/txt/',
    test_image_rdir='test_OBB/images/',
    test_label_rdir='test_OBB/txt/',
    data_pipeline='obb_{}_800x800_pipeline',
    num_classes=20
)

HRSC2016=dict(
    version='',
    data_name='HRSC2016_txt_Dataset',
    data_root='/home/zzc/data/ObjDet/HRSC2016/processed/',
    train_image_rdir='train/images/',
    train_label_rdir='train/txt/',
    test_image_rdir='test/images/',
    test_label_rdir='test/txt/',
    data_pipeline='obb_{}_800x800_pipeline',
    num_classes=1
)
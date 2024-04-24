class Config:
    device = 'cpu'

    # Label config
    charset = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    max_seq_len = 10
    
    # Training hyperparams
    epochs = 100
    batch_size = 2
    lr = 0.0001
    
    # Model configs
    latent = True # always
    img_size = (64, 256)
    timestep_embedding_dim = 320
    inference_timestep = 50
    pretrained_sd = '/data/ocr/namvt17/WordStylist/diffusers/v1-5-pruned-emaonly.ckpt'
    
    # Misc
    dataset_root = '/data/ocr/namvt17/WordStylist/data'
    label_path = 'gt/gan.iam.tr_va.gt.filter27'
    save_path = './output/'
    
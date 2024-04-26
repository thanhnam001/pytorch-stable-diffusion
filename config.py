class LabelStyleEncoder(object):
    d_model = 512
    nhead = 8
    num_encoder_layers = 2
    num_head_layers = 1
    dec_layers = 2
    dim_feedforward = 2048
    dropout = 0.1
    activation = "relu"
    normalize_before = True
    num_writers = 339

class Config(object):
    device = 'cuda:3'

    # Label config
    charset = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz' # 52
    max_seq_len = 10
    
    # Training hyperparams
    epochs = 100
    batch_size = 32
    lr = 0.0001
    
    # Model configs
    latent = True # always
    img_size = (64, 256)
    timestep_embedding_dim = 320
    inference_timestep = 50
    # Weights
    pretrained_sd = '/data/ocr/namvt17/WordStylist/diffusers/v1-5-pruned-emaonly.ckpt'
    use_pretrained_encoder = True # VAE encoder
    use_pretrained_diffuser = False # Diffuser UNET
    use_pretrained_decoder = True # VAE decoder
    use_pretrained_text_encoder = False # CLIP
    # LabelStyleEncoder
    label_style_encoder = LabelStyleEncoder
    
    # Misc
    dataset_root = '/data/ocr/namvt17/WordStylist/data'
    label_path = 'gt/gan.iam.tr_va.gt.filter27'
    # label_path = 'gt/train_samples'
    save_path = './output'
    
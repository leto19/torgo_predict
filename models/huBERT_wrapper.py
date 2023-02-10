from torch import Tensor, nn
import fairseq
import torch
import torch.nn.functional as F
import speechbrain as sb

#import matplotlib.pyplot as plt

class HuBERTWrapper_extractor(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        ckpt_path = "models/facebook/HuBERT/hubert_base_ls960.pt"
        models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
        self.model = models[0].feature_extractor
        
    def forward(self, data: Tensor):
        #print(self.model)
        #print(data.shape)
        return self.model(data)


class HuBERTWrapper_full(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        ckpt_path = "models/facebook/HuBERT/hubert_base_ls960.pt"

        models = fairseq.checkpoint_utils.load_model_ensemble([ckpt_path])
        full_model = models[0][0]
        full_model.features_only =True
        self.model = full_model
        
    

    def forward(self, data: Tensor):
        
        """
        my_output = None
        def my_hook(module_,input_,output_):
            nonlocal my_output
            my_output = output_

        a_hook = self.model.encoder.layers[6].final_layer_norm.register_forward_hook(my_hook)
        self.model(data)
        a_hook.remove()
        """
        
        my_output =self.model(data)
        return my_output['x']

class PoolAttFF(torch.nn.Module):
    '''
    PoolAttFF: Attention-Pooling module with additonal feed-forward network.
    '''         
    def __init__(self, dim_head_in):
        super().__init__()
        
        self.linear1 = nn.Linear(dim_head_in, 2*dim_head_in)
        self.linear2 = nn.Linear(2*dim_head_in, 1)
        
        self.linear3 = nn.Linear(dim_head_in, 1)
        
        self.activation = F.relu
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x: Tensor):

        att = self.linear2(self.dropout(self.activation(self.linear1(x))))
        att = att.transpose(2,1)
        att = F.softmax(att, dim=2)
        x = torch.bmm(att, x) 
        x = x.squeeze(1)
        
        x = self.linear3(x)
        
        return x  
class PoolAttFFMulti(torch.nn.Module):
    '''
    PoolAttFF: Attention-Pooling module with additonal feed-forward network.
    '''         
    def __init__(self, dim_head_in):
        super().__init__()
        
        self.linear1 = nn.Linear(dim_head_in, 2*dim_head_in)
        self.linear2 = nn.Linear(2*dim_head_in, 1)
        
        self.linear3 = nn.Linear(dim_head_in, 1)
        
        self.activation = F.relu
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x: Tensor):

        att = self.linear2(self.dropout(self.activation(self.linear1(x))))
        att = att.transpose(2,1)
        att = F.softmax(att, dim=2)
        x = torch.bmm(att, x)
        x = x.squeeze(1)

        x = self.linear3(x)

        return x 

class SimpleFF(nn.Module):
    def __init__(self, dim_head_in):
        super().__init__()
        
        self.linear1 = nn.Linear(dim_head_in, dim_head_in//2)
        self.linear2 = nn.Linear(dim_head_in//2,  dim_head_in//4)
        
        self.linear3 = nn.Linear(dim_head_in//4, 1)
        
        self.activation = F.relu
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x: Tensor):

        print(x.shape)
        #x = torch.mean(x,(1, 2))
        print("after mean",x.shape)
        x = self.dropout(self.activation(self.linear1(x)))
        print(x.shape)

        x = self.dropout(self.activation(self.linear2(x)))
        print(x.shape)
        

        x = self.dropout(self.activation(self.linear3(x)))
        print(x.shape)


        #for l in [self.linear1,self.linear2,self.linear3]:
        #    x = self.dropout(self.activation(l(x)))

        
        return x




class MetricPredictorMulti(nn.Module):
    """Metric estimator for enhancement training.

    Consists of:
     * four 2d conv layers
     * channel averaging
     * three linear layers

    Arguments
    ---------
    kernel_size : tuple
        The dimensions of the 2-d kernel used for convolution.
    base_channels : int
        Number of channels used in each conv layer.
    """

    def __init__(
        self, dim_extractor=512, hidden_size=512//2, activation=nn.LeakyReLU,
    ):
        super().__init__()

        self.activation = activation(negative_slope=0.3)

        #self.BN = nn.BatchNorm1d(num_features=1, momentum=0.01)


        self.feat_extract = HuBERTWrapper_extractor()
        self.lin1 = nn.Linear(512,249)
        
        self.conv1 = nn.Conv2d(1,64,(5,5))
        self.conv2 = nn.Conv2d(64,1,(5,5))
        
        self.blstm = nn.LSTM(
            input_size=249,
            hidden_size=hidden_size,
            num_layers=2,
            dropout=0.1,
            bidirectional=True,
            batch_first=True,
        )
        self.lin2 = nn.Linear(512,256)
        
        self.attenPool1 = PoolAttFFMulti(dim_extractor//2)
        self.attenPool2 = PoolAttFFMulti(dim_extractor//2)
        self.attenPool3 = PoolAttFFMulti(dim_extractor//2)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x,spec):
        #print("----- IN THE MODEL -----")
        #print(x.shape)
        #out = self.BN(x)
        
        out_feats = self.feat_extract(x).permute(0,2,1)
        
        spec_conv = self.conv2(self.conv1(spec))

        feats_lin = self.lin1(out_feats)

        #print(feats_lin.shape)
        #print(feats_lin)
        combin = torch.cat((spec_conv,feats_lin),1)
        """
        print("combin",combin.shape)
        plt.imshow(combin.squeeze().detach().cpu().numpy().T)
        plt.savefig("out.png")
        plt.close()
        """
        out,_ = self.blstm(combin)
        """
        print("out",out.shape)
        plt.imshow(out.squeeze().detach().cpu().numpy().T)
        plt.savefig("out2.png")
        plt.close()
        """
        out = self.lin2(out)
        #input(">>>")
        #out = out_feats
        out1 = self.attenPool1(out)
        out1 = self.sigmoid(out1)
        
        out2 = self.attenPool2(out)
        out2 = self.sigmoid(out2)
        
        out3 = self.attenPool3(out)
        out3 = self.sigmoid(out3)
        #print(out1.shape,out2.shape,out3.shape)
        out = torch.cat([out1,out2,out3],dim=1)
        #out = self.sigmoid(out)
        
        return out,out_feats


class MetricPredictorSimple(nn.Module):
    """Metric estimator for enhancement training.

    Consists of:
     * four 2d conv layers
     * channel averaging
     * three linear layers

    Arguments
    ---------
    kernel_size : tuple
        The dimensions of the 2-d kernel used for convolution.
    base_channels : int
        Number of channels used in each conv layer.
    """

    def __init__(
        self, dim_extractor=512, hidden_size=512//2, activation=nn.LeakyReLU,
    ):
        super().__init__()
        self.feat_extract = HuBERTWrapper_extractor()
        self.FFLayers = SimpleFF(dim_extractor)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #print("----- IN THE MODEL -----")
        #print(x.shape)
        #out = self.BN(x)
        
        out_feats = self.feat_extract(x).permute(0,2,1)
        print(out_feats.shape)
        #out,_ = self.blstm(out_feats)
        out = out_feats
        out = self.FFLayers(out)
        #out = self.sigmoid(out)
        #print("----- LEAVING THE MODEL -----")

        return out,out_feats

class MetricPredictor(nn.Module):
    """Metric estimator for enhancement training.

    Consists of:
     * four 2d conv layers
     * channel averaging
     * three linear layers

    Arguments
    ---------
    kernel_size : tuple
        The dimensions of the 2-d kernel used for convolution.
    base_channels : int
        Number of channels used in each conv layer.
    """

    def __init__(
        self, dim_extractor=512, hidden_size=512//2, activation=nn.LeakyReLU,
    ):
        super().__init__()

        self.activation = activation(negative_slope=0.3)

        #self.BN = nn.BatchNorm1d(num_features=1, momentum=0.01)


        self.feat_extract = HuBERTWrapper_extractor()

        
        self.blstm = nn.LSTM(
            input_size=dim_extractor,
            hidden_size=hidden_size,
            num_layers=2,
            dropout=0.1,
            bidirectional=True,
            batch_first=True,
        )
        
        
        self.attenPool = PoolAttFF(dim_extractor)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #print("----- IN THE MODEL -----")
        #print(x.shape)
        #out = self.BN(x)
        
        out_feats = self.feat_extract(x).permute(0,2,1)
        #print(out_feats.shape)
        out,_ = self.blstm(out_feats)
        #out = out_feats
        out = self.attenPool(out)
        out = self.sigmoid(out)
        #print("----- LEAVING THE MODEL -----")

        return out,out_feats

class MetricPredictorFull(nn.Module):
    """Metric estimator for enhancement training.

    Consists of:
     * four 2d conv layers
     * channel averaging
     * three linear layers

    Arguments
    ---------
    kernel_size : tuple
        The dimensions of the 2-d kernel used for convolution.
    base_channels : int
        Number of channels used in each conv layer.
    """

    def __init__(
        self, dim_extractor=768, hidden_size=768//2, activation=nn.LeakyReLU,
    ):
        super().__init__()

        self.activation = activation(negative_slope=0.3)

        #self.BN = nn.BatchNorm1d(num_features=1, momentum=0.01)


        self.feat_extract = HuBERTWrapper_full()
        
        
        self.blstm = nn.LSTM(
            input_size=dim_extractor,
            hidden_size=hidden_size,
            num_layers=2,
            dropout=0.1,
            bidirectional=True,
            batch_first=True,
        )
        
        
        self.attenPool = PoolAttFF(dim_extractor)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #print("----- IN THE MODEL -----")
        #print(x.shape)
        #out = self.BN(x)
        
        out_feats = self.feat_extract(x)#.permute(0,2,1)
        out,_ = self.blstm(out_feats)
        #out = out_feats
        out = self.attenPool(out)
        out = self.sigmoid(out)
        #print("----- LEAVING THE MODEL -----")

        return out,out_feats

if __name__ == "__main__":
    import soundfile as sf
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    from speechbrain.nnet.losses import mse_loss 
    import os
    import csv
    import speechbrain as sb
    from speechbrain.processing.features import spectral_magnitude,STFT
    from scipy.special import expit
    #PATH="results/lstm-hubert-ench-fine-100-2700/save/CKPT+2022-10-30+16-58-24+00/predictor.ckpt"
    PATH="results/lstm-hubert-ench-fine-100-8000/save/CKPT+2022-10-30+15-46-53+00/predictor.ckpt"
    mod = MetricDiscriminator_NISQA()
    mod.load_state_dict(torch.load(PATH))
    mod.eval()
    try:
        from models.will_utils import channel_sort
    except:
        from will_utils import channel_sort
    def compute_feats(wavs):
        """Feature computation pipeline"""
        stft = STFT(hop_length=16,win_length=32,sample_rate=16000,n_fft=512,window_fn=torch.hamming_window)
        feats = stft(wavs)
        feats = spectral_magnitude(feats, power=0.5)
        feats = torch.log1p(feats)
        return feats 


    def get_img(in_rep,ax,fig):
        im = ax.imshow(sigmoid_v(in_rep.detach().cpu().numpy()),aspect='auto',norm='linear')
        fig.colorbar(im,cmap='plasma')

    def sigmoid(x):
        return 1 / (1 + expit(-x))
    sigmoid_v =  np.vectorize(sigmoid)
    clean_root= "/fastdata/acp20glc/VoiceBank/clean_testset_wav_16k/"
    noisy_root = "/fastdata/acp20glc/VoiceBank/noisy_testset_wav_16k/"
    
    
    
    f_names = os.listdir(clean_root)
    with open("/fastdata/acp20glc/projects/MetricGANdiscrim_feats/log_testset.csv") as f:
        csv_reader = csv.reader(f)

    
        

        hs_list = []
        ef_list = []
        next(csv_reader,None)
        for row in csv_reader:
            file, loc, snr = row
            clean_wav = sb.dataio.dataio.read_audio(clean_root+file+".wav").unsqueeze(0)
            noisy_wav = sb.dataio.dataio.read_audio(noisy_root+file+".wav").unsqueeze(0)
            #noisy_wav = np.random.random(clean_wav.shape)
            """
            if len(clean_wav) > len(clean_wav_ref):
                clean_wav = clean_wav[:len(clean_wav_ref)]
            else:
                clean_wav_ref = clean_wav_ref[:len(clean_wav)]
            """
            #noisy_wav = clean_wav_ref
            pred_score,feat_rep = mod(clean_wav)
            print(pred_score)
            fig,ax = plt.subplots()

            ax = get_img(feat_rep[0,:,:].T,ax,fig)
            plt.savefig("tuned_rep.png")
            plt.close()
            #plt.show()
            mod2 = MetricDiscriminator_NISQA()
            pred_score,feat_rep = mod2(clean_wav)
            print(pred_score)
            fig,ax = plt.subplots()

            ax = get_img(feat_rep[0,:,:].T,ax,fig)
            plt.savefig("base_rep.png")
            exit()
            print(clean_wav.shape,noisy_wav.shape)
            clean_rep = mod(clean_wav)

            noisy_rep = mod(noisy_wav)

            #extract_feats_mse = mse_loss(clean_rep,noisy_rep).detach().cpu().numpy()
            #extract_feats_cosine = np.mean(torch.nn.CosineSimilarity()(clean_rep,noisy_rep).detach().cpu().numpy())



            ###################################################################
            
            fig, ax = plt.subplots(2,2,figsize=(3.378,3))
            font = {'family':'times new roman','size'   : 15}

            plt.rc('font', **font)
             
            ax[0,0].imshow(compute_feats(clean_wav).T.flipud().squeeze(0),aspect='auto')
            #plt.xlabel(r'$\mathbf{S}_\mathrm{SG}$')
            ax[0,0].set_xlabel(r'$T$')
            ax[0,0].set_ylabel(r'$F_{Hz}$')
            ax[0,0].set_title(r'$\mathbf{S}_\mathrm{SG}$')
            ax[0,1].imshow(compute_feats(noisy_wav).T.flipud().squeeze(0),aspect='auto')
            ax[0,1].set_xlabel(r'$\mathbf{S}_\mathrm{SG}$')
            ax[0,1].set_xlabel(r'$T$')
            ax[0,1].set_ylabel(r'$F_{Hz}$')
            ax[0,1].set_title(r'$\mathbf{X}_\mathrm{SG}$')
            
            clean_rep_disp = clean_rep.T.squeeze().detach().cpu().numpy()
            print(clean_rep_disp.shape)
            chan_idx_clean = channel_sort(clean_rep_disp.T)
            clean_rep_disp = clean_rep_disp[:,chan_idx_clean]
            print(clean_rep_disp.shape)

            noisy_rep_disp = noisy_rep.T.squeeze().detach().cpu().numpy()
            noisy_rep_disp = noisy_rep_disp[:,chan_idx_clean]
            print(clean_rep_disp.shape)
            ax[1,0].imshow(np.fliplr(clean_rep_disp).T,cmap="plasma",aspect='auto')

            ax[1,0].set_ylabel(r'$F$')
            ax[1,0].set_xlabel(r'$T$')
            ax[1,0].set_title(r'HuBERT $\mathbf{S}_\mathrm{FE}$',fontsize=15)

            #plt.subplot(144)
            ax[1,1].imshow(np.fliplr(noisy_rep_disp).T,cmap="plasma",aspect='auto')
            ax[1,1].set_ylabel(r'$F$')
            ax[1,1].set_xlabel(r'$T$')
            ax[1,1].set_title(r'HuBERT $\mathbf{X}_\mathrm{FE}$')

       
            plt.tight_layout()
            plt.show()
            plt.savefig(file+"_hubert.png")
            plt.close()
            #input(">>>")           

    """
    for wav in ["p232_025.wav","p232_020.wav"]:
        audio_data,fs = get_audio(wav,16000)
        audio_data_np = np.float32(audio_data)
        audio_data_pt = torch.from_numpy(audio_data_np)
        #print(mod)
        print(audio_data_pt.shape)
        rep = mod(audio_data_pt)
        #print(rep.shape)
        #print(rep)
        print(rep["last_hidden_state"],rep["last_hidden_state"].shape)
        print(rep["extract_features"],rep["extract_features"].shape)

        hidden_state = rep["last_hidden_state"]
        feats = rep["extract_features"]
        
        fig, axs = plt.subplots(3)
        axs[0].plot(audio_data_np)
        axs[1].imshow(feats.squeeze().detach().numpy())
        axs[2].imshow(hidden_state.squeeze().detach().numpy())

        plt.savefig("out_%s.png"%wav.split(".")[0])
    """
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn as nn
import speechbrain as sb
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import datetime
import torchinfo
#word error rate
import jiwer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def compute_feats(wavs,fs):
        """Feature computation pipeline"""
        resample_rate = 16000
        #sum stero channels 
        #resample to 16000 for input to HASPI
        wavs = wavs[:,:,0]
        #print(wavs.shape)
        #resampler = T.Resample(fs, resample_rate, dtype=wavs.dtype).to("cuda:0")
        #resampled_waveform = resampler(wavs)

        return wavs

def audio_pipeline(path,fs=32000):
    audio = sb.dataio.dataio.read_audio_multichannel(path)    
    return audio 
def audio_pipeline_len(path,fs=32000):
    audio = sb.dataio.dataio.read_audio_multichannel(path)
    return audio.shape[0]/fs 

def format_correctness(y):
    #convert correctness percentage to tensor
    y = torch.tensor([y])
    
    return y


def test_model(model,test_data,optimizer,criterion):
    out_list = []
    model.eval()
    path_list = test_data["wav_path"]
    correctness_list = test_data["correctness"]


    running_loss = 0.0
    loss_list = []
   
    test_dict = {}
    i = 0
    for path,corr in zip(path_list,correctness_list):
        test_dict["file_%s"%i] = {"wav_path":path,"correctness":corr}
        i+=1
    #print(train_dict)
    dynamic_items = [
         {"func": lambda l: format_correctness(l),
        "takes": "correctness",
        "provides": "formatted_correctness"},
        {"func": lambda l: audio_pipeline(l,32000),
        "takes": "wav_path",
        "provides": "wav"},
        #{"func": lambda l: audio_pipeline("%s/clarity_data/HA_outputs/train/%s_HL-output.wav"%(DATAROOT,l),44100),
        #"takes": "signal",
        #"provides": "wav"},
        #{"func": lambda l: audio_pipeline("%s/clarity_data/scenes//%s_target_anechoic.wav"%(DATAROOT,l),44100),
        #"takes": "scene",
        #"provides": "clean_wav"},
    ]
    test_set = sb.dataio.dataset.DynamicItemDataset(test_dict,dynamic_items)
    #train_set.set_output_keys(["wav","clean_wav", "formatted_correctness","audiogram_np","haspi"])
    test_set.set_output_keys(["wav", "formatted_correctness"])

    my_dataloader = DataLoader(test_set,1,collate_fn=sb.dataio.batch.PaddedBatch)
    print("starting testing/validation...")
    for batch in tqdm(my_dataloader):
        batch = batch.to("cuda:0")
        wavs,correctness = batch
        correctness =correctness.data
        wavs_data = wavs.data
        #print("wavs:%s\n correctness:%s\n"%(wavs.data.shape,correctness))
        target_scores = correctness
   
        
        feats = compute_feats(wavs_data,16000) 
        #print(feats.shape)
       
        output,_ = model(feats.float())
        

        loss = criterion(output, target_scores)
        #for x1,y1 in zip(output.detach().cpu().numpy(),target_scores.cpu().detach().numpy()):
        #    print("P: %s | T: %s"%(x1,y1))
        out_list.append(output.detach().cpu().numpy()[0][0])

        loss_list.append(loss.item())
        # print statistics
        running_loss += loss.item()
    print("Average testing/validation MSE loss: %s"%(sum(loss_list)/len(loss_list)))

    return out_list,sum(loss_list)/len(loss_list)




    



def train_model(model,train_data,optimizer,criterion):
    model.train()
    path_list = train_data["wav_path"]
    correctness_list = train_data["correctness"]



    #print(name_list)
    #columns_titles = ["signal",'scene', 'listener', 'system', 'mbstoi', 'correctness', 'predicted']
    #train_data = train_data.reindex(columns_titles)
    #train_data = train_data.to_dict()
    running_loss = 0.0
    loss_list = []
   
    train_dict = {}
    i = 0
    for path,corr in zip(path_list,correctness_list):
        train_dict["file_%s"%i] = {"wav_path":path,"correctness":corr}
        i+=1
    #print(train_dict)
    dynamic_items = [
         {"func": lambda l: format_correctness(l),
        "takes": "correctness",
        "provides": "formatted_correctness"},
        {"func": lambda l: audio_pipeline(l,16000),
        "takes": "wav_path",
        "provides": "wav"},
        #{"func": lambda l: audio_pipeline_len(l,16000),
        #"takes": "wav_path",
        #"provides": "length"},
        #{"func": lambda l: audio_pipeline("%s/clarity_data/HA_outputs/train/%s_HL-output.wav"%(DATAROOT,l),44100),
        #"takes": "signal",
        #"provides": "wav"},
        #{"func": lambda l: audio_pipeline("%s/clarity_data/scenes//%s_target_anechoic.wav"%(DATAROOT,l),44100),
        #"takes": "scene",
        #"provides": "clean_wav"},
    ]
    train_set = sb.dataio.dataset.DynamicItemDataset(train_dict,dynamic_items)#.filtered_sorted(key_max_value={'length':10})

    train_set.set_output_keys(["wav", "formatted_correctness"])

    my_dataloader = DataLoader(train_set,10,collate_fn=sb.dataio.batch.PaddedBatch)
    print("starting training...")
    for batch in tqdm(my_dataloader):
        batch = batch.to("cuda:0")
        wavs,correctness = batch
        correctness =correctness.data
        wavs_data = wavs.data
        #print("wavs:%s\n correctness:%s\n"%(wavs.data.shape,correctness))
        target_scores = correctness
        #target_scores = [sum(get_mean(haspi)[0])/2]
        
        #print(wavs_data.shape)
        feats = compute_feats(wavs_data,16000) 
        #print(feats.shape)
       
        #input(">>>")
        optimizer.zero_grad()
        output,_ = model(feats.float())
       
        #for x1,y1 in zip(output.detach().cpu().numpy(),target_scores.cpu().detach().numpy()):
        #    print("P: %s | T: %s"%(x1,y1))
        #print(output,scores)
        loss = criterion(output,target_scores)
        #print(loss)
        #don't update the SSSR model parameters 
        #for name, param in model.named_parameters():
        #    if "feat_extract" in name:
        #        param.requires_grad = False
        loss.backward()
        optimizer.step()
        loss_list.append(loss)
        running_loss += loss.item()
    print("Average training loss: %s"%(sum(loss_list)/len(loss_list)))
    
    return model,optimizer,criterion






def save_model(model,opt,epoch,args,val_loss):
    p = args.model_dir
    if not os.path.exists(p):
        os.mkdir(p)
    m_name = "%s-%s"%(args.model,args.seed)
    torch.save(model.state_dict(),"%s/%s_%s_%s_model.pt"%(p,m_name,epoch,val_loss))
    torch.save(opt.state_dict(),"%s/%s_%s_%s_opt.pt"%(p,m_name,epoch,val_loss))


def find_length(wav_path):
    wav = audio_pipeline(wav_path,16000)
    return len(wav)/16000


def main(args):

    # Load the  data
    df_intel = pd.read_json(args.in_json_file)
    data = df_intel.T
    data["predicted"] = np.nan  # Add column to store intel predictions
    data["wav_path"] = data.index
    #print(data["words"],type(data["words"]))
    #print(data["words"].tolist())
    # create the session and speaker columns
    data["session"] = data["wav_path"].apply(lambda x: x.split("/")[-3])
    data["speaker"] = data["wav_path"].apply(lambda x: x.split("/")[-4])
    # compute the word count of the transcript and the length of the utterance
    data["word_count"] = data["transcript"].apply(lambda x: len(x.split(" ")))
    data["length"] = data["wav_path"].apply(lambda x: find_length(x))

    # filter out the utterances with only one word
    if args.remove_one_word:
        data = data[data["word_count"] > 1]
    # filter out the utterances with length > 10 seconds
    data = data[data["length"] < 10]
    #compute the word error rate
    wer = []
    cer = []
    for w,t in zip(data["words"].tolist(), data["transcript"].tolist()):
        wer.append(jiwer.wer(w,t))
        cer.append(jiwer.cer(w,t))

    #create the correctness column
    if args.correctness_type == "CER":
        data["correctness"] = 1-np.array(wer)
    elif args.correctness_type == "WER":
        data["correctness"] = 1-np.array(cer)
    else:
        print("Correctness type not recognised")
        exit(1)

    # set the correctness to 0 if it is negative
    data.loc[data["correctness"] < 0, "correctness"] = 0
    
    #we can optionally turn the task into binary classification
    # i.e if any of the words were recognised
    #data.loc[data["correctness"] > 0, "correctness"] = 1
    
    # plot the correctness distribution
    plt.hist(data["correctness"])
    plt.title("Correctness distribution")
    plt.xlabel("Correctness")
    plt.ylabel("Count")
    plt.savefig("correctness.png")
    plt.close()

    # split the data into train, val and test
    test_speakers = args.test_speakers
    test_data = data[data["speaker"].isin(test_speakers)]
    train_data = data[~data["speaker"].isin(test_speakers)]
    train_data,val_data = train_test_split(train_data,test_size=0.1)
    print("Trainset: %s\nValset: %s\nTestset: %s "%(train_data.shape,val_data.shape,test_data.shape))
    # shuffle the training data
    train_data = train_data.sample(frac=1).reset_index(drop=True)
    
    # use this line for testing :) 
    #train_data = train_data[:50]
    
    #set up the torch objects
    print("creating model: %s"%args.model)
    torch.manual_seed(args.seed)
    if args.model == "XLSREncoder":
        from models.ni_predictors import XLSRMetricPredictorEncoder
        model = XLSRMetricPredictorEncoder().to("cuda:0")
    elif args.model == "XLSRFull":
        from models.ni_predictors import XLSRMetricPredictorFull
        model = XLSRMetricPredictorFull().to("cuda:0")
    elif args.model == "HuBERTEncoder":
        from models.ni_predictors import HuBERTMetricPredictorEncoder
        model = HuBERTMetricPredictorEncoder().to("cuda:0")
    elif args.model == "HuBERTFull":
        from models.ni_predictors import HuBERTMetricPredictorFull
        model = HuBERTMetricPredictorFull().to("cuda:0")
    elif args.model == "Spec":
        from models.ni_predictors import SpecMetricPredictor
        model = SpecMetricPredictor().to("cuda:0")
    
    #set save location:
    today = datetime.date.today()
    date = today.strftime("%H-%M-%d-%b-%Y")

    model_dir = "save/%s_%s_%s_%s"%(args.model,args.correctness_type,date,args.seed)
    args.model_dir = model_dir

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(),lr=args.lr)
    #print the model summary
    torchinfo.summary(model)
    #train the model
    print("-----------------")
    print("Starting training of model: %s\nobjective: %s\nlearning rate: %s\nseed: %s\nepochs %s\nsave location: %s/"%(args.model,args.correctness_type,args.lr,args.seed,args.n_epochs,args.model_dir))
    print("-----------------")
    
    for epoch in range(args.n_epochs):
        print("Epoch: %s"%(epoch))
        model,optimizer,criterion = train_model(model,train_data,optimizer,criterion)
        

        #get predictions for the val set 
        predictions,val_loss = test_model(model,val_data,optimizer,criterion)
        save_model(model,optimizer,epoch,args,val_loss)
    
        print("-----------------")
    
    #load the model with the lowest validation loss
    
    print(model_dir)
    model_files = os.listdir(model_dir)

    model_files = [x for x in model_files if "model" in x]
    print(model_files)
    model_files.sort(key=lambda x: float(x.split("_")[-2].strip(".pt")))
    model_file = model_files[0]
    print("Loading model: %s"%model_file)
    model.load_state_dict(torch.load("%s/%s"%(model_dir,model_file)))
    
    predictions,_ = test_model(model,test_data,optimizer,criterion)
    test_data["predicted"] = predictions

    print(test_data[["correctness", "predicted"]])
    print(test_data["correctness"].corr(test_data["predicted"]))
    test_data[["wav_path","correctness", "predicted"]].to_csv(args.out_csv_file, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #REQUIRED
    parser.add_argument(
        "in_json_file", help="JSON file containing the metadata of the dataset"
    )
    parser.add_argument(
        "out_csv_file", help="output csv file containing the correctness predictions"
    )
    #OPTIONAL
    parser.add_argument(
        "--correctness_type", help="correctness type: WER or CER", default="CER",required=False
    )
    parser.add_argument(
        "--n_epochs", help="number of epochs", default=30, type=int
    )
    parser.add_argument(
        "--lr", help="learning rate", default=0.001, type=float
    )
    parser.add_argument(
        "--model", help="model type" , default="XLSREncoder",
    )
    parser.add_argument(
        "--seed", help="torch seed" , default=1234,
    )
    parser.add_argument(
        "--remove_one_word", help="remove one word utterances" , default=True,
    )
    parser.add_argument(
        "--test_speakers", help="test speakers" , default=["F01","M01"],type=list
    )
    args = parser.parse_args()
    print(args)
    main(args)

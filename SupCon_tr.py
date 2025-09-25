# -*- coding: utf-8 -*-
import numpy as np
import time
from pathlib import Path
import sys
sys.path.append("..")

from model_utils import get_model, TimmResNetWrapper
import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
torch.cuda.empty_cache()

from temp_schedulers import GCosineTemperatureSchedulerM, M_NegCosineTemperatureScheduler, LinearScheduler, ExponentialIncreaseScheduler, RandomScheduler, LogarithmicIncreaseScheduler, LinearDecreasingScheduler
from losses import SupConLoss
import random
from schedulers import get_scheduler
import models 
from model_util import save_model, load_checkpoint
from data.open_set_datasets import get_class_splits, get_datasets
from torch.utils.data import DataLoader
import evaluation
from sklearn.neighbors import KNeighborsClassifier
from LabelSmoothing import smooth_cross_entropy_loss
import json 
import argparse
from set_default_params import set_default_forSupCon, set_filename_forSupCon

def fetch_args():

    parser = argparse.ArgumentParser("Training")

    # Dataset
    parser.add_argument('--dataset', type=str, default='cifar-10-10', help="cifar-10-10|cifar-10-100|tinyimagenet|cub")
    parser.add_argument('--out_num', type=int, default=50, help='For cifar-10-100')
    parser.add_argument('--image_size', type=int, default=32)
    parser.add_argument('--N_closed', type=int, default=None)

    # optimization
    parser.add_argument('--optim', type=str, default='sgd', help="Which optimizer to use {adam, sgd}")
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=None, help="learning rate for model")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="LR regularisation on weights")
    parser.add_argument('--epochs', type=int, default=600)
    parser.add_argument('--load_epoch', type=int, default=None)
    parser.add_argument('--scheduler', type=str, default='cosine_warm_restarts_warmup')
    parser.add_argument('--num_restarts', type=int, default=2, help='How many restarts for cosine_warm_restarts schedule')

    # model
    parser.add_argument('--encoder_lossname', type=str, default='SupCon',help="SupCon|SimCLR")
    parser.add_argument('--classifier_lossname', type=str, default='CE')
    parser.add_argument('--label_smoothing', type=float, default=None, help="Smoothing constant for classifier label smoothing."
                                                                        "No smoothing if None or 0")
    
    parser.add_argument('--architecture', type=str, default='VGG32')
    parser.add_argument('--resnet50_pretrain', type=str, default='places_moco',
                        help='Which pretraining to use if --model=timm_resnet50_pretrained.'
                             'Options are: {iamgenet_moco, places_moco, places}', metavar='BOOL')
    parser.add_argument('--feat_dim', type=int, default=128, help="Feature vector dim, only for classifier32 at the moment")

    # aug
    parser.add_argument('--transform', type=str, default='rand-augment')
    parser.add_argument('--rand_aug_m', type=int, default=6)
    parser.add_argument('--rand_aug_n', type=int, default=1)
    parser.add_argument('--alpha',type=float, default = 1.0)

    # misc
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--split_train_val', default=False, type=bool,
                        help='Subsample training set to create validation set', metavar='BOOL')

    parser.add_argument('--PRINT_INTERVAL', type=int, default=10)
    parser.add_argument('--SAVE_EPOCH', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--split_idx', default=0, type=int, help='0-4 OSR splits for each dataset')



    parser.add_argument('--supcon_label_smoothing', type=bool, default=False)
    parser.add_argument('--supcon_alpha', type=float, default=0.2, help='smoothing for SupCon')
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--temperature_scheduling', type=bool, default=False)
    parser.add_argument('--temp_scheduler', type=str, default='cos', help = 'cos|gcos|gcosm|Hcos|step|ecos|sin')
    parser.add_argument('--T', type=int, default=200)
    parser.add_argument('--shift', type=float, default=0.0, help='shift for gcos= 0,+-0.25,+-0.5,+-0.75,+1')
    parser.add_argument('--Tp', type=float, default=0.4)
    parser.add_argument('--K', type=int, default=100)

    args = parser.parse_args()
    args = set_default_forSupCon(args)
    print(args)
    return args


def check_gpus():
  DEVICE = torch.device('cpu')
  num_gpus=0
  if torch.cuda.is_available():
    DEVICE = torch.device('cuda')  # torch.device('cpu')
    # Get the number of available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {num_gpus}")
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
  return DEVICE,num_gpus


def seed_torch(seed=1029):

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def fetch_checkpoint(model,opt,scheduler,model_dir,model_filename,args):
   """checking whether a previous check point exists"""
   print("checking whether a previous check point exists ...")

   check_epoch = args.SAVE_EPOCH
   model_filepath = model_dir + model_filename + '_' + str(check_epoch)+ '.pt'
   if(os.path.isfile(model_filepath)):
      while(os.path.isfile(model_filepath)):
        check_epoch = check_epoch + args.SAVE_EPOCH
        model_filepath = model_dir + model_filename + '_'+ str(check_epoch)+ '.pt'
    
      if(args.load_epoch is None):
         args.load_epoch = check_epoch - args.SAVE_EPOCH
    
      model_filepath = model_dir + model_filename + '_'+ str(args.load_epoch)+ '.pt'
      s_epoch,  model, opt,scheduler = load_checkpoint(model_filepath,model,opt,scheduler)

   else:
      s_epoch= 1
      print("starting from scratch ....")
   return s_epoch, model, opt,scheduler

def fetch_datasets(args):
    if(args.dataset in ['cub','aircraft','scars']):
        args.train_classes, args.open_set_classes = get_class_splits(args.dataset, args.split_idx,cifar_plus_n=args.out_num, cub_osr='split')
    else:
        args.train_classes, args.open_set_classes = get_class_splits(args.dataset, args.split_idx,cifar_plus_n=args.out_num)

    if(args.dataset=='cub' and args.N_closed!=100):
        args.train_classes = args.train_classes[:args.N_closed]
    if(args.dataset=='aircraft' and args.N_closed!=50):
        args.train_classes = args.train_classes[:args.N_closed]
    if(args.dataset=='scars' and args.N_closed!=98):

        args.train_classes = args.train_classes[:args.N_closed]

    out_loader_easy = None
    if(args.dataset in ['cub','aircraft','scars']):
       args.open_set_classes2 = args.open_set_classes['Easy']
       args.open_set_classes = args.open_set_classes['Hard'] + args.open_set_classes['Medium']
       datasets_easy = get_datasets(args.dataset, transform=args.transform, train_classes=args.train_classes,
                                open_set_classes=args.open_set_classes2, balance_open_set_eval=False,
                                split_train_val=args.split_train_val, image_size=args.image_size, seed=args.seed,
                                args=args)
       out_loader_easy = DataLoader(datasets_easy['test_unknown'], batch_size=args.batch_size,
                                        shuffle=False, sampler=None, num_workers=args.num_workers)


    datasets = get_datasets(args.dataset, transform=args.transform, train_classes=args.train_classes,
                                open_set_classes=args.open_set_classes, balance_open_set_eval=False,
                                split_train_val=args.split_train_val, image_size=args.image_size, seed=args.seed,
                                args=args)

    dataloaders = {}
    for k, v, in datasets.items():
       shuffle = True if k == 'train' else False
       dataloaders[k] = DataLoader(v, batch_size=args.batch_size,
                                        shuffle=shuffle, sampler=None, num_workers=args.num_workers)


    train_loader = dataloaders['train']
    test_loader = dataloaders['val']
    out_loader = dataloaders['test_unknown']
    return train_loader, test_loader,out_loader,out_loader_easy



def extract_all_features(model,loader,from_head=False,normalize=False):
      model.eval()
      z_list=None
      y_list=None
      for idx,(x,y,idd) in enumerate(loader):
        if(isinstance(x,list)):
            x = x[0]
        x = x.cuda()
        y = y.cuda()
        
        if(from_head):
          z = model(x)
        else:
          z = model.encoder(x,normalize)

        z = z.cpu().detach().numpy()
        y = y.cpu().detach().numpy()
        z_list =  z if z_list is None else np.concatenate((z_list,z),axis=0)
        y_list =  y if y_list is None else np.concatenate((y_list,y),axis=0)
      return z_list,y_list


def train_knn(model,train_loader,K=100,from_head=False,normalize=True):

    num_train=len(train_loader.dataset)
    neigh = KNeighborsClassifier(n_neighbors=K)
    model.eval()

    z_list,y_list=extract_all_features(model,train_loader,from_head=from_head,normalize=normalize)

    neigh.fit(z_list,y_list)

    return neigh



def accuracy_from_knns(model,KNN,test_loader,K=100,from_head=False,normalize=True,individual=False,method='dist',plot_mis_samples=False):
    model.eval()
    num_test = len(test_loader.dataset)
    f =None
    p=None
    y_list =None
    
    running_correct = 0
    for idx,(x,y,idd) in enumerate(test_loader):
       #if(idx==0):
       if(isinstance(x,list)):
         x = x[0]
       x=x.cuda()
       y=y.cuda()
       f_x =None
       p_x= None
       
       if(from_head):
            z = model(x)
       else:
            z = model.encoder(x,normalize = normalize)

       z = z.cpu().detach().numpy()


       f_x = KNN.predict(z)
       if(method=='prob'):
         p_x = np.max(KNN.predict_proba(z),axis=1)
       elif(method=='dist'):
         distt,_= KNN.kneighbors(z,n_neighbors=K,return_distance=True)
         p_x = np.exp(-np.max(distt,axis=1))

       y = y.cpu().detach().numpy()
       correct = y==f_x
       running_correct += np.sum(y == f_x)
       f = f_x if f is None else np.concatenate((f,f_x),axis=0)
       p = p_x if p is None else np.concatenate((p,p_x),axis=0)
       y_list = y if y_list is None else np.concatenate((y_list,y),axis=0)
    
    acc = running_correct / num_test
    prob_osr=list(p) 
    return acc,prob_osr,y_list,f



def get_performance_OSR(model,train_loader,test_loader,out_loader,K=100,from_head=False,normalize=True,method='dist'):

   print("KNN based inference: -----")
   knn = train_knn(model,train_loader,K=K,from_head=from_head,normalize=normalize)

   #print("test")
   acc_test,probOSR_test,y_list,predicted_label = accuracy_from_knns(model,knn,test_loader,K=K,from_head=from_head,normalize=normalize,method=method)
   #print("out")
   _,probOSR_out,_,_ = accuracy_from_knns(model,knn,out_loader,K=K,from_head=from_head,normalize=normalize,method=method)

   results={}
   prob_known = np.array(probOSR_test)
   prob_unknown = np.array(probOSR_out)
   results_ = evaluation.metric_ood(prob_known, prob_unknown)['Bas']
   OSCR = evaluation.compute_oscr(prob_known,prob_unknown, np.array(predicted_label), np.array(y_list))

   results['AUROC']=results_['AUROC']
   results['ACC']=acc_test
   results['OSCR']=OSCR
   return results



def train(model,train_loader,test_loader,out_loader,args):

    if(args.temperature_scheduling):
        if(args.temp_scheduler == 'M_NegCos'):
          TS= M_NegCosineTemperatureScheduler(tau_plus=args.Tp,tau_minus=args.temperature,T=args.T)
        if(args.temp_scheduler=='gcosm'):
          TS= GCosineTemperatureSchedulerM(tau_plus=args.Tp,tau_minus=args.temperature,T=args.T,shift=args.shift)
        if(args.temp_scheduler in ['random']):
          TS = RandomScheduler(tau_plus=args.Tp,tau_minus=args.temperature)
        if(args.temp_scheduler == 'linear'):
          TS = LinearScheduler(tau_plus=args.Tp,tau_minus=args.temperature)
        if(args.temp_scheduler == 'exp'):
          TS = ExponentialIncreaseScheduler(tau_plus=args.Tp,tau_minus=args.temperature)
        if(args.temp_scheduler == 'log'):
          TS = LogarithmicIncreaseScheduler(tau_plus=args.Tp,tau_minus=args.temperature)
        if(args.temp_scheduler == 'lineardecrease'):
          TS = LinearDecreasingScheduler(tau_plus=args.Tp,tau_minus=args.temperature)


 
    criterion = SupConLoss(temperature=args.temperature,label_smoothing=args.supcon_label_smoothing,alpha=args.supcon_alpha,num_class=args.N_closed)
    
    torch.cuda.empty_cache()
    train_loss = []
    for epoch in range(args.s_epoch,args.epochs+1):
      if(args.temperature_scheduling):
        criterion.temperature = TS.get_temperature(epoch)

      model.train()
      running_loss= 0
      running_corrects = 0
      for idx,(x,y,idd) in enumerate(train_loader):

        start_time = time.time()
        y = y.cuda()
        x = torch.cat([x[0], x[1]], dim=0)
        x = x.cuda()
        bsz = y.shape[0]
        opt.zero_grad()
        features = model(x, normalize=True)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        loss = criterion(features, y)
        loss.backward()
        opt.step()
        
        # statistics
        running_loss += loss.item() * x.size(0)

      scheduler.step() 
      train_loss = running_loss / len(train_loader.dataset)
      print(
            "Epoch: {:03d} Train Loss: {:.3f}"
            .format(epoch, train_loss))
      
      
      if(epoch % args.PRINT_INTERVAL ==0):
         print("Starting inferencing ...")
         results=get_performance_OSR(model,train_loader,test_loader,out_loader,K=args.K,from_head=True)
         print("Performance after {:03d} epochs: --- Accuracy: {:.4f}, AUC: {:.4f}.".format(epoch,results['ACC'],results['AUROC']))

      if(epoch % args.SAVE_EPOCH ==0):
            save_model(model, opt, scheduler, epoch, model_dir, model_filename)
            print("model saved")


def test_linear(model,classifier,testloader,from_softmax = False):
      torch.cuda.empty_cache()
      train_loss = []
      correct, total = 0, 0
      
      model.eval()
      classifier.eval()
      f = None
      p = None
      y_list = None
    
      running_loss= 0
      running_corrects = 0
      for idx,(x,y,idd) in enumerate(testloader):

        start_time = time.time()
        if(isinstance(x,list)):
            x = x[0]

        y = y.cuda()
        x = x.cuda()
        bsz = y.shape[0]

        with torch.set_grad_enabled(False):
          loss = None
          features =[]

          feature = model.encoder(x,normalize=False)
          logit = classifier(feature.detach())
          if(from_softmax):
                logit = torch.nn.Softmax(dim = -1)(logit)
          predictions = logit.data.max(1)[1]
        

          max_logit = np.max(logit.data.cpu().numpy(), axis = 1)
          #print(logit)
          f_x=predictions.data.cpu().numpy() #np.expand_dims(predictions.data.cpu().numpy(),axis=1)
          p_x =max_logit #np.expand_dims(max_logit,axis=1)


        y = y.cpu().detach().numpy()
        correct = y==f_x
        f = f_x if f is None else np.concatenate((f,f_x),axis=0)
        p = p_x if p is None else np.concatenate((p,p_x),axis=0)
        y_list = y if y_list is None else np.concatenate((y_list,y),axis=0)

        # statistics
        running_corrects += np.sum(f_x==y)

      prob_osr= list(p) 
      acc = running_corrects / len(testloader.dataset)
      return acc,prob_osr,y_list,f




def get_performance_OSR_Linear(model,train_loader,test_loader,out_loader,from_softmax =False):

   print("Max logit based inference: -----")

   acc_test,probOSR_test,y_list,predicted_label=test_linear(model,classifier,test_loader, from_softmax =from_softmax)
   #print("out")
   _,probOSR_out,_,_=test_linear(model,classifier,out_loader, from_softmax =from_softmax)


   results={}
   prob_known = np.array(probOSR_test)
   prob_unknown = np.array(probOSR_out)
   results_ = evaluation.metric_ood(prob_known, prob_unknown)['Bas']
   OSCR = evaluation.compute_oscr(prob_known,prob_unknown,np.array(predicted_label), np.array(y_list))
   results['AUROC']=results_['AUROC']
   results['ACC']=acc_test
   results['OSCR']=OSCR


   print("End of inference --------------")
   return results


def train_linear(model,classifier,train_loader,test_loader,out_loader,args):
    if(args.label_smoothing ==None):
        criterion = nn.CrossEntropyLoss()
    torch.cuda.empty_cache()
    train_loss = []

    for epoch in range(args.s_epoch,args.epochs+1):
      model.eval()
      classifier.train()

      running_loss= 0
      running_corrects = 0
      for idx,(x,y,idd) in enumerate(train_loader):
    
        start_time = time.time()
        if(isinstance(x,list)):
            x = x[0]

        y = y.cuda()
        x = x.cuda()
        bsz = y.shape[0]

        opt.zero_grad()
        loss = None
        features=[]
        with torch.no_grad():
             feature = model.encoder(x,normalize=False)
          
        output = classifier(feature.detach())
        if(args.label_smoothing == None):
           loss = criterion(output,y)
        else:
            loss = smooth_cross_entropy_loss(output, y, args.label_smoothing)
        loss.backward()
        opt.step()
    
        # statistics
        running_loss += loss.item() * x.size(0)

      scheduler.step()
      train_loss = running_loss / len(train_loader.dataset)

      print(
            "Epoch: {:03d} Train Loss: {:.3f}"
            .format(epoch, train_loss))

      if(epoch % args.SAVE_EPOCH == 0):
         print("Starting inferencing ...")
         results=get_performance_OSR_Linear(model,train_loader,test_loader,out_loader)
         print("Performance after {:03d} epochs: --- Accuracy: {:.4f}, AUC: {:.4f}.".format(epoch,results['ACC'],results['AUROC']))
        

if __name__ == '__main__':
  args = fetch_args()

  model_dir = 'models/'
  if not os.path.exists(model_dir):
    os.makedirs(model_dir)
  if not os.path.exists('results/'):
    os.makedirs('results/')

  model_filename, result_filename = set_filename_forSupCon(args)
  print(model_filename)
  print(result_filename)

  seed_torch(seed=args.seed)

  model = models.create_model(args)
  DEVICE,args.num_gpus=check_gpus()
  model.to(DEVICE)

  opt = optim.SGD(model.parameters(),
                          lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)


  scheduler = get_scheduler(opt, args)

  s_epoch, model, opt,scheduler = fetch_checkpoint(model,opt,scheduler,model_dir,model_filename,args)
  args.s_epoch = s_epoch

  pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
  print("The total number of trainable model parameters: ", pytorch_total_params)

  train_loader, test_loader,out_loader,out_loader_easy = fetch_datasets(args)

  train(model,train_loader,test_loader,out_loader,args)
  save_model(model,opt,scheduler, args.epochs, model_dir, model_filename)

  
  print("---------- training Linear Classifier -----------")
  classifier = models.LinearClassifier(num_classes = args.N_closed,feat_dim=args.feat_dim)
  classifier = classifier.cuda()

  opt = optim.SGD(classifier.parameters(),
                          lr=args.lr,
                          momentum=0.9,
                          weight_decay=1e-4)

  args.epochs = 200
  args.num_restarts = 1
  scheduler = get_scheduler(opt, args)
  args.s_epoch = 1
  model_filename = model_filename + '_LinearHead'
  model_filepath = model_dir + model_filename + f'_{args.epochs}.pt'

  train_linear(model,classifier,train_loader,test_loader,out_loader,args)
  save_model(classifier,opt,scheduler, args.epochs, model_dir, model_filename)

  all_results={}
  inference_method='clshead_maxlogit'
  print(inference_method)
  results=get_performance_OSR_Linear(model,train_loader,test_loader,out_loader)
  print(results)
  all_results[inference_method]=results
    
  if(args.dataset in ['cub','aircraft','scars']):
      inference_method='clshead_maxlogit_easy'
      print(inference_method)
      results=get_performance_OSR_Linear(model,train_loader,test_loader,out_loader_easy)
      print(results)
      all_results[inference_method]=results


  print(all_results)
  with open(result_filename, 'w') as f:
       json.dump(all_results, f) 

  print("Results Saved!!!!")

import os 
import torch

def save_model(model, OPT, scheduler, EPOCH,  model_dir,  model_filename):

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_filepath = model_dir + model_filename +'_' +str(EPOCH)+ '.pt'
    torch.save({
            'epoch':EPOCH,
            'model_state_dict':model.state_dict(),
            'optimizer_state_dict':OPT.state_dict(),
            'scheduler_state_dict':scheduler.state_dict(),
            }, model_filepath)



"""loading a previous check point"""
def load_checkpoint(model_filepath,model,opt=None,scheduler=None):
  
    checkpoint = torch.load(model_filepath)
    s_epoch= checkpoint['epoch']+1 #starting epoch for training

    model.load_state_dict(checkpoint['model_state_dict'])
    if(opt is not None):
      opt.load_state_dict(checkpoint['optimizer_state_dict'])
    if(scheduler is not None):
      scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    print("previous checkpoint found at epoch: ",s_epoch-1)

    return s_epoch,  model, opt,scheduler 


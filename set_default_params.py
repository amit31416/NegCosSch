

def set_default_forCE(args):
  if(args.dataset == 'tinyimagenet'):
     args.image_size =64
     if(args.N_closed is None):
       args.N_closed = 20
     args.out_num = 180
     args.seed = 0
     args.rand_aug_m = 9
     args.num_workers= 8
  if(args.lr is None):
     if(args.dataset=='tinyimagenet'):
       args.lr = 0.01
     else:
       args.lr=0.1

  if(args.dataset=='cifar-10-100'):
    if(args.N_closed is None):
      args.N_closed = 4
  if(args.dataset=='cifar-10-10'):
    if(args.N_closed is None):
       args.N_closed = 6
    args.out_num = 4
  if(args.dataset=='cub'):
    if(args.N_closed is None):
      args.N_closed = 100
  if(args.dataset=='aircraft'):
    if(args.N_closed is None):
      args.N_closed = 50
  if(args.dataset=='scars'):
    if(args.N_closed is None):
      args.N_closed = 98
  if(args.dataset=='cifar-100-10'):
    if(args.N_closed is None):
      args.N_closed = 100
      args.out_num = 10
  args.img_size = args.image_size
  return args

def set_filename_forCE(args):
  model_filename = f'{args.dataset}_{args.N_closed}class_split{args.split_idx}_{args.architecture}_CE_lr{args.lr}'

  if(args.batch_size in [128,12]):
    model_filename = model_filename + f'_B{args.batch_size}'
  if(args.temperature!=1.0 and args.temperature_scheduling==False):
    model_filename = model_filename + f'_temp{args.temperature}'
  if(args.temperature_scheduling):
    model_filename = model_filename + f'_TS{args.temp_scheduler}{args.Tp}to{args.temperature}'
    if(args.temp_scheduler in ['gcos','gcosm']):
       model_filename = model_filename + f's{args.shift}'
    model_filename =model_filename + f'_T{args.T}'
  if(args.label_smoothing is not None):
    model_filename = model_filename + f'_LS{args.label_smoothing}'

  model_filename = f'{args.seed}_' + model_filename


  if(args.dataset=='cifar-10-100'):
    pre=model_filename.split('cifar-10-100')[0]
    post = model_filename.split('cifar-10-100')[-1]
    result_filename = 'results/'+ pre + f'{args.dataset}-{args.out_num}' + post + '.json'
  elif(args.dataset in ['cub','aircraft','scars']):
    result_filename = 'results/'+model_filename+'_hard.json'
  else:
    result_filename = 'results/'+model_filename+'.json'

  return model_filename, result_filename

def set_default_forSupCon(args):
   if(args.dataset == 'tinyimagenet'):
       args.image_size =64
       if(args.N_closed is None):
          args.N_closed = 20
       args.out_num = 180
       args.seed = 0
       args.rand_aug_m = 9
       args.num_workers= 8
   if(args.lr is None):
       if(args.dataset=='tinyimagenet' and args.encoder_lossname in ['SupCon','ProtoCon','SimCLR','BT']):
           args.lr = 0.05
       else:
           args.lr=0.1
   if(args.dataset=='cifar-10-100'):
      if(args.N_closed is None):
         args.N_closed = 4
   if(args.dataset=='cifar-10-10'):
      if(args.N_closed is None):
         args.N_closed = 6
      args.out_num = 4
   if(args.dataset=='cub'):
      if(args.N_closed is None):
         args.N_closed = 100
   if(args.dataset=='aircraft'):
      if(args.N_closed is None):
         args.N_closed = 50
   if(args.dataset=='scars'):
      if(args.N_closed is None):
         args.N_closed = 98

   args.img_size = args.image_size
   return args


def set_filename_forSupCon(args):
   model_filename = f'{args.dataset}_{args.N_closed}class_split{args.split_idx}_{args.architecture}_{args.encoder_lossname}_lr{args.lr}'

   if(args.batch_size==128):
      model_filename = model_filename + f'_B{args.batch_size}'

   if(args.temperature!=0.1 and args.temperature_scheduling==False):
      model_filename = model_filename + f'_temp{args.temperature}'

   if(args.temperature_scheduling):
      model_filename = model_filename + f'_TS{args.temp_scheduler}{args.Tp}to{args.temperature}'
      if(args.temp_scheduler in ['gcos','gcosm']):
         model_filename = model_filename + f's{args.shift}'
      model_filename =model_filename + f'_T{args.T}'
   if(args.supcon_label_smoothing):
      model_filename = model_filename + f'_LSalpha{args.supcon_alpha}'

   model_filename = f'{args.seed}_' + model_filename

   if(args.dataset=='cifar-10-100'):
      pre=model_filename.split('cifar-10-100')[0]
      post = model_filename.split('cifar-10-100')[-1]
      result_filename = 'results/'+ pre + f'{args.dataset}-{args.out_num}' + post + '.json'
   elif(args.dataset in ['cub','aircraft','scars']):
      result_filename = 'results/'+model_filename+'_hard.json'
   else:
      result_filename = 'results/'+model_filename+'.json'

   return model_filename, result_filename


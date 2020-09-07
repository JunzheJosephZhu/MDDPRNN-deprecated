import os                                                                       
import tqdm                                                                     
in_dir = 'wsj0_sph'
all_file = os.walk(in_dir)
out_dir = 'wsj0'                                                                               
listOfFiles = list()                                                            
for (dirpath, dirnames, filenames) in all_file:                                 
  listOfFiles += [os.path.join(dirpath, file) for file in filenames]            
                                                                                
for tmp in tqdm.tqdm(listOfFiles):                                              
  try:                                                                          
    tmp1, tmp2 = tmp.split('.')                                                 
  except:                                                                       
    pass                                                                        
  if tmp2 == 'wv1':
    newfile = tmp1.replace(in_dir, out_dir) + '.wav'
    if not os.path.exists(os.path.dirname(newfile)):
        os.makedirs(os.path.dirname(newfile))
                                                        
    command = 'tools/sph2pipe_v2.5/sph2pipe -f wav  %s %s' % (tmp, newfile)          
    os.system(command)    

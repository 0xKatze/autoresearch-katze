import sys, os, faulthandler
faulthandler.enable()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
torch.cuda.is_available = lambda: True
sys.argv = ['eval','--seeds','42','--cap','10','--T','300','--device','cuda','--max-folds','1',
            '--attacks','OM2C-nb10']
print('>>> OM2C-only GPU eval (expandable_segments)', flush=True)
import eval_pg_floor_agnncert_fair as E
E.main()
print('>>> OK', flush=True)

import sys, os, faulthandler
faulthandler.enable()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["CUDA_VISIBLE_DEVICES"]=""
import torch
torch.cuda.is_available = lambda: False
torch.set_num_threads(4)
sys.argv = ['eval','--seeds','42','--cap','10','--T','300','--device','cpu','--max-folds','1',
            '--attacks','OM2C-nb10,FM-nb10-eb1-lownorm,FM-nb10-full']
print('>>> starting eval main()', flush=True)
import eval_pg_floor_agnncert_fair as E
E.main()
print('>>> eval main() returned OK', flush=True)

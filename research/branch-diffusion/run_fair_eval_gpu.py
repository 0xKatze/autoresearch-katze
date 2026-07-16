import sys, os, faulthandler
faulthandler.enable()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
# torch.cuda.is_available() segfaults at import on this host; bypass the C-level probe with a
# constant True so real cuda ops (.cuda(), GCN forward) still run on the GPU.
torch.cuda.is_available = lambda: True
sys.argv = ['eval', '--seeds', '42', '--cap', '10', '--T', '300', '--device', 'cuda', '--max-folds', '1',
            '--attacks', 'OM2C-nb10,FM-nb10-eb1-lownorm,FM-nb10-full']
print('>>> GPU eval starting', flush=True)
import eval_pg_floor_agnncert_fair as E
E.main()
print('>>> GPU eval returned OK', flush=True)

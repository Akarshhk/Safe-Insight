from app import network_guard
import traceback
orig = network_guard._record_violation
def patch(*args):
  if args[0] == b'huggingface.co':
    print('VIOLATION', args, 'ACTIVE:', network_guard._sanctioned_download_active, 'GLOBALSANCT:', network_guard._global_sanctioned_domains)
  orig(*args)
network_guard._record_violation = patch

from app import network_guard
import logging
orig = network_guard._record_violation
def debug_violation(*args, **kwargs):
  import traceback; traceback.print_stack()
  print('DEBUG VIOLATION:', args, 'active:', network_guard._sanctioned_download_active)
  orig(*args, **kwargs)
network_guard._record_violation = debug_violation

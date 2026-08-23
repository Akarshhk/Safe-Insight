from app import network_guard
network_guard._sanctioned_download_active = True
network_guard._global_sanctioned_domains = {'huggingface.co', '.huggingface.co', '.hf.co'}
print('Result:', network_guard._is_local_address(b'huggingface.co'))

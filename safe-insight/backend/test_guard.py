from app import network_guard
import contextvars
network_guard._sanctioned_domains.set({'huggingface.co', '.huggingface.co', '.hf.co'})
print(network_guard._is_local_address(b'huggingface.co'))

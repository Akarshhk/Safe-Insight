import contextvars
var = contextvars.ContextVar('x', default=set())
print(id(var.get()))
def t():
  print(id(var.get()))
import threading
threading.Thread(target=t).start()

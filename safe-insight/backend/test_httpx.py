import contextvars, httpx, asyncio, socket
var=contextvars.ContextVar('test', default=None)
orig = socket.getaddrinfo
def mock_getaddrinfo(*args, **kwargs):
  print('GETADDRINFO VAR:', var.get())
  return orig(*args, **kwargs)
socket.getaddrinfo = mock_getaddrinfo
async def main():
  token = var.set('yes')
  try:
    async with httpx.AsyncClient() as c:
      await c.get('http://example.com')
  finally:
    var.reset(token)
asyncio.run(main())

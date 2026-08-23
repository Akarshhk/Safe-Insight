import uvicorn, fastapi, contextvars, httpx, asyncio, anyio
var=contextvars.ContextVar('test', default='no')
app=fastapi.FastAPI()
@app.get('/test')
async def test():
  async def gen():
    var.set('yes')
    async with httpx.AsyncClient() as c:
      def work(): return var.get()
      yield await anyio.to_thread.run_sync(work) + '\n'
  return fastapi.responses.StreamingResponse(gen())
if __name__=='__main__': uvicorn.run(app, port=8766)

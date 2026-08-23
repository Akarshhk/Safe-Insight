import anyio, contextvars
var=contextvars.ContextVar('test', default='no')
async def main():
    var.set('yes')
    def work(): return var.get()
    print(await anyio.to_thread.run_sync(work))
anyio.run(main)

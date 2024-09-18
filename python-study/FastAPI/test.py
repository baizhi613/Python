# import uvicorn
# from fastapi import FastAPI
# app = FastAPI()
# @app.get("/")
# async def root():
#     return {"message": "Hello World"}
#
# if __name__ == '__main__':
#     config = uvicorn.Config(app, host='0.0.0.0', port=8009)
#     server = uvicorn.Server(config)
#     await server.serve()
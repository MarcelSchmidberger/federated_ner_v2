async def sendWsMessage(data):
    async with websockets.connect('ws://' + gatewayWsUrl) as websocket:
        await websocket.send(json.dumps(data))
        message = await websocket.recv()
        return json.loads(message)
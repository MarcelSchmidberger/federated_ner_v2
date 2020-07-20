import syft as sy
from syft.serde import protobuf
from syft_proto.execution.v1.plan_pb2 import Plan as PlanPB
from syft_proto.execution.v1.state_pb2 import State as StatePB
from syft.grid.clients.static_fl_client import StaticFLClient
from syft.execution.state import State
from syft.execution.placeholder import PlaceHolder
from syft.execution.translation import TranslationTarget

import torch as th
from torch import nn

import os
import websockets
import json
import requests

import jwt
auth_token = jwt.encode({}, private_key, algorithm='RS256').decode('ascii')

print(auth_token)

# Make authentication request:
auth_request = {
    "type": "federated/authenticate",
    "data": {
        "model_name": name,
        "model_version": version,
        "auth_token": auth_token,
    }
}
auth_response = await sendWsMessage(auth_request)
print('Auth response: ', json.dumps(auth_response, indent=2))

# Make the cycle request:
cycle_request = {
    "type": "federated/cycle-request",
    "data": {
        "worker_id": auth_response['data']['worker_id'],
        "model": name,
        "version": version,
        "ping": 1,
        "download": 10000,
        "upload": 10000,
    }
}
cycle_response = await sendWsMessage(cycle_request)
print('Cycle response:', json.dumps(cycle_response, indent=2))

worker_id = auth_response['data']['worker_id']
request_key = cycle_response['data']['request_key']
model_id = cycle_response['data']['model_id'] 
training_plan_id = cycle_response['data']['plans']['training_plan']

# Model
req = requests.get(f"http://{gatewayWsUrl}/federated/get-model?worker_id={worker_id}&request_key={request_key}&model_id={model_id}")
model_data = req.content
pb = StatePB()
pb.ParseFromString(req.content)
model_params_downloaded = protobuf.serde._unbufferize(hook.local_worker, pb)
print("Params shapes:", [p.shape for p in model_params_downloaded.tensors()])

# Plan "list of ops"
req = requests.get(f"http://{gatewayWsUrl}/federated/get-plan?worker_id={worker_id}&request_key={request_key}&plan_id={training_plan_id}&receive_operations_as=list")
pb = PlanPB()
pb.ParseFromString(req.content)
plan_ops = protobuf.serde._unbufferize(hook.local_worker, pb)
#print(plan_ops.code)
#print(plan_ops.torchscript)

# Plan "torchscript"
req = requests.get(f"http://{gatewayWsUrl}/federated/get-plan?worker_id={worker_id}&request_key={request_key}&plan_id={training_plan_id}&receive_operations_as=torchscript")
pb = PlanPB()
pb.ParseFromString(req.content)
plan_ts = protobuf.serde._unbufferize(hook.local_worker, pb)
#print(plan_ts.code)
#print(plan_ts.torchscript.code)

# Plan "tfjs"
req = requests.get(f"http://{gatewayWsUrl}/federated/get-plan?worker_id={worker_id}&request_key={request_key}&plan_id={training_plan_id}&receive_operations_as=tfjs")
pb = PlanPB()
pb.ParseFromString(req.content)
plan_tfjs = protobuf.serde._unbufferize(hook.local_worker, pb)
print(plan_tfjs.code)
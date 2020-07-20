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

from model import Net, set_model_params, naive_sgd, softmax_cross_entropy_with_logits
from preprocessing import read_files, prepare_embeddings, text_to_indices, pad_sentences


sy.make_hook(globals())
# force protobuf serialization for tensors
hook.local_worker.framework = None
th.random.manual_seed(1)

train_data_path = "/home/marcel/voize/voize-react-native/nlp/train"

train_sentences = read_files([os.path.join(train_data_path, p) for p in os.listdir(train_data_path)])
word_embeddings, word2Idx, label2Idx, idx2Label = prepare_embeddings(train_sentences,"../embeddings/german.model" )
X,Y = text_to_indices(train_sentences, word2Idx, label2Idx)
X,Y = pad_sentences(X,Y, word2Idx, label2Idx)



model = Net(th.FloatTensor(word_embeddings))


@sy.func2plan()
def training_plan(X, y, batch_size, lr, model_params):
    # inject params into model
    set_model_params(model, model_params)

    # forward pass
    logits = model.forward(X)
    
    # loss
    loss = softmax_cross_entropy_with_logits(logits, y, batch_size)

    # backprop
    loss.backward()

    # step
    updated_params = [
        naive_sgd(param, lr=lr)
        for param in model_params
    ]
    
    # accuracy
    pred = th.argmax(logits, dim=1)
    target = th.argmax(y, dim=1)
    acc = pred.eq(target).sum().float() / batch_size

    return (
        loss,
        acc,
        *updated_params
    )

# Dummy input parameters to make the trace
model_params = [param.data for param in model.parameters()]  # raw tensors instead of nn.Parameter
lr = th.tensor([0.01])
batch_size = th.tensor([3.0])

_ = training_plan.build(X, Y, batch_size, lr, model_params, trace_autograd=True)


training_plan.base_framework = TranslationTarget.TENSORFLOW_JS.value
training_plan.base_framework = TranslationTarget.PYTORCH.value

@sy.func2plan()
def avg_plan(avg, item, num):
    new_avg = []
    for i, param in enumerate(avg):
        new_avg.append((avg[i] * num + item[i]) / (num + 1))
    return new_avg

# Build the Plan
_ = avg_plan.build(model_params, model_params, th.tensor([1.0]))

async def sendWsMessage(data):
    async with websockets.connect('ws://' + gatewayWsUrl) as websocket:
        await websocket.send(json.dumps(data))
        message = await websocket.recv()
        return json.loads(message)

# Default gateway address when running locally 
gatewayWsUrl = "127.0.0.1:5000"
grid = StaticFLClient(id="test", address=gatewayWsUrl, secure=False)
grid.connect()# These name/version you use in worker
import time;
name = "mnist2"
version = "1.0.0"

client_config = {
    "name": name,
    "version": version,
    "batch_size": 64,
    "lr": 0.005,
    "max_updates": 100  # custom syft.js option that limits number of training loops per worker
}

server_config = {
    "min_workers": 5,
    "max_workers": 5,
    "pool_selection": "random",
    "do_not_reuse_workers_until_cycle": 6,
    "cycle_length": 28800,  # max cycle length in seconds
    "num_cycles": 5,  # max number of cycles
    "max_diffs": 1,  # number of diffs to collect before avg
    "minimum_upload_speed": 0,
    "minimum_download_speed": 0,
    "iterative_plan": True  # tells PyGrid that avg plan is executed per diff
}
from authentication import public_key
server_config["authentication"] = {
    "type": "jwt",
    "pub_key": public_key,
}

model_params_state = State(
    state_placeholders=[
        PlaceHolder().instantiate(param)
        for param in model_params
    ]
)

response = grid.host_federated_training(
    model=model_params_state,
    client_plans={'training_plan': training_plan},
    client_protocols={},
    server_averaging_plan=avg_plan,
    client_config=client_config,
    server_config=server_config
)

print("Host response:", response)
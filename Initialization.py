def initialization(model):
    state_dict = model.state_dict()
    for name, param in state_dict.items():
        print("before")
        print(param)
        print(name)
        tt = param + 0.9
        state_dict[name].copy_(tt)
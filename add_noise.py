def add_noise(model, big_epsilon, small_epsilon):
    for name, param in model.state_dict().items():
        if name == "hiddenLayer.weight":
            param = param.add(big_epsilon)
            print(param)
        elif name == "outPut2.weight":
            param = param.add(small_epsilon)
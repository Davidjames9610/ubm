import my_torch.tuts2.torch_transforms as torch_t

# for overused
def file_vad_normalize():
    return torch_t.ComposeTransform([
        torch_t.FileToTensor(),
        torch_t.NormalizeSox(),
        torch_t.SileroVad()
    ])

def file_normalize():
    return torch_t.ComposeTransform([
        torch_t.FileToTensor(),
        torch_t.NormalizeSox(),
    ])

def file_to_numpy():
    return torch_t.ComposeTransform([
        torch_t.FileToTensor(),
        torch_t.NormalizeSox(),
        torch_t.SileroVad(),
        torch_t.TensorToNumpy()
    ])




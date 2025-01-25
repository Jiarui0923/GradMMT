class ModelMultiTrack(object):
    
    def __init__(self, model, tracks=[
        [('encoderA', ['dropout']), ('decoderA', ['dropout'])]
    ]):
        self.forward_data = {}
        self.backward_data = {}
        self.track_configs = tracks
        self.model = model
        self.track_named_modules = {}
        for line_id, line in enumerate(tracks):
            for module_name, submodules in line:
                if module_name not in self.track_named_modules:
                    self.track_named_modules[module_name] = []
                self.track_named_modules[module_name].append([line_id, submodules])
        
        self.tracked_modules = []
        
        self._forward_hooks = []
        self._backward_hooks = []
        for name, module in self.model.named_modules():
            module_name = name.split('.')[-1]
            module_parents = name.split('.')[:-1]
            _main_module = self.get_last_union(self.track_named_modules, module_parents)
            if _main_module is not None:
                for line_id, submodules in self.track_named_modules[_main_module]:
                    for submodule in submodules:
                        if submodule == name[-len(submodule):]:
                            module_id = id(module)
                            self.tracked_modules.append([module_id, line_id])
                            self._forward_hooks.append(module.register_forward_hook(self.forward_hook))
                            self._backward_hooks.append(module.register_full_backward_hook(self.backward_hook))
                        
    
    @staticmethod
    def get_last_union(base_list, query_list):
        for i in reversed(query_list):
            if i in base_list: return i
        return None
    
    def reset(self):
        self.forward_data = {}
        self.backward_data = {}
                
    def forward_hook(self, module, input, output):
        self.forward_data[id(module)] = output.cpu()
        
    def backward_hook(self, module, input, output):
        self.backward_data[id(module)] = input[0].cpu()
        
    @property
    def forwards(self):
        lines_data = [[] for _ in self.track_configs]
        for (module_id, line_id) in reversed(self.tracked_modules):
            lines_data[line_id].append(self.forward_data[module_id])
        return lines_data

    @property
    def backwards(self):
        lines_data = [[] for _ in self.track_configs]
        for (module_id, line_id) in reversed(self.tracked_modules):
            lines_data[line_id].append(self.backward_data[module_id])
        return lines_data

    def clear_hook(self):
        for hook in self._backward_hooks: hook.remove()
        for hook in self._forward_hooks: hook.remove()
        
    
    def criterion(self, output, groundtruths): raise NotImplemented
    
    def __call__(self, groundtruths, *args, **kwargs):
        self.reset()
        output = self.model(*args, **kwargs)
        self.model.zero_grad()
        loss = self.criterion(output, groundtruths)
        loss.backward()
        return output
        

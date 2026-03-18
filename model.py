"""
Implementation of YOLOv3 architecture
"""

import torch
import torch.nn as nn

""" 
Information about architecture config:
Tuple is structured by (filters, kernel_size, stride) 
Every conv is a same convolution. 
List is structured by "B" indicating a residual block followed by the number of repeats
"S" is for scale prediction block and computing the yolo loss
"U" is for upsampling the feature map and concatenating with a previous layer
"""
import config
class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bn_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not bn_act, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)
        self.use_bn_act = bn_act

    def forward(self, x):
        if self.use_bn_act:
            return self.leaky(self.bn(self.conv(x)))
        else:
            return self.conv(x)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, pad, groups, batch_normalize, activation):
        super(ConvBlock, self).__init__()

        padding = (kernel_size - 1)//2 if pad else 0

        bias = False if batch_normalize else True
        
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups=groups,
            bias=bias
        )
        
        self.bn = nn.BatchNorm2d(out_channels) if batch_normalize else nn.Identity()

        if activation == 'relu':
            self.activ = nn.ReLU(inplace=True)
        elif activation == 'linear':
            self.activ = nn.Identity()
        else:
            raise ValueError("bruh choose Relu or Linear activation")
    
    def forward(self,x):
        return self.activ(self.bn(self.conv(x)))

class ShortcutBlock(nn.Module):
    def __init__(self, from_layer, activation):
        super(ShortcutBlock, self).__init__()
        self.from_layer = from_layer

        if activation == 'relu':
            self.activ = nn.ReLU(inplace=True)
        elif activation == 'linear':
            self.activ = nn.Identity()
        else:
            raise ValueError("bruh choose Relu or Linear activation")
    def forward(self,x_current, x_previous):
        return self.activ(x_current + x_previous)

class RouteBlock(nn.Module):
    def __init__(self):
        super(RouteBlock, self).__init__()
        
class MaxPoolBlock(nn.Module):
    def __init__(self,kernel_size,stride):
        super(MaxPoolBlock,self).__init__()
        if stride == 1:
            padding = (kernel_size - 1)//2
        else:
            padding = 0
        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride,padding=padding)
    def forward(self,x):
        return self.pool(x)

class UpsampleBlock(nn.Module):
    def __init__(self,stride):
        super(UpsampleBlock,self).__init__()
        self.stride = stride
    def forward(self,x):
        return nn.functional.interpolate(x, scale_factor=self.stride, mode = 'nearest')

class YoloBlock(nn.Module):
    def __init__(self, anchors, mask, num_classes, scale_x_y=1.0):
        super(YoloBlock,self).__init__()
        self.anchors = [(anchors[i],anchors[i+1]) for i in range(0, len(anchors),2)]
        self.anchors = [self.anchors[i] for i in mask]
        
        self.num_classes = num_classes
        self.num_anchors = len(self.anchors)
        self.scale_x_y = scale_x_y

        self.bbox_attrs = 5 + self.num_classes

    def forward(self, x):
        # x shape: [batch_size, 255, grid_size, grid_size]
        batch_size = x.size(0)
        grid_size = x.size(2)
        device = x.device

        # reshape tensor to: [batch_size, 3 anchors, 85 attr, grid_size, grid_size]
        prediction = x.view(batch_size, self.num_anchors, self.bbox_attrs, grid_size, grid_size)

        # rearange shape to: [batch_size, 3 anchors, grid_size, grid_size, 85 attr]
        prediction = prediction.permute(0,1,3,4,2).contiguous()

        output = torch.cat((
            prediction[..., 4:5],   # Confidence at index 0
            prediction[..., 0:4],   # Bounding box coords at indices 1, 2, 3, 4
            prediction[..., 5:]     # Classes at indices 5+
        ), dim=-1)

        return output

def parse_config(path):
    with open(path, 'r') as f:
        lines = f.read().split('\n')
        lines = [x for x in lines if x and not x.startswith('#')]
        lines = [x.rstrip().lstrip() for x in lines]

        module_defs = []
        for line in lines:
            if line.startswith('['):  # This marks the start of a new block
                module_defs.append({})
                module_defs[-1]['type'] = line[1:-1].rstrip()
                if module_defs[-1]['type'] == 'convolutional':
                    module_defs[-1]['batch_normalize'] = 0  # Set default value for batch_normalize
            else:
                key, value = line.split('=')
                module_defs[-1][key.rstrip()] = value.lstrip()

    return module_defs

class MyMOLO(nn.Module):
    def __init__(self, config_path):
        super(MyMOLO, self).__init__()
        self.module_defs = parse_config(config_path)
        self.hyperparams, self.module_list = self._create_modules(self.module_defs)
    def _create_modules(self,module_defs):
        hyperparams = module_defs.pop(0) # [net] block
        output_filters = [int(hyperparams.get('channels',3))]
        module_list = nn.ModuleList()
        for i, module_def in enumerate(module_defs):
            modules = nn.Sequential()
            
            if module_def["type"] == 'convolutional':
                bn = int(module_def.get("batch_normalize",0))
                filters = int(module_def['filters'])
                kernel_size = int(module_def['size'])
                stride = int(module_def.get('stride',1))
                pad = int(module_def.get('pad',0))
                groups = int(module_def.get('groups',1))
                activation = module_def['activation']
                modules.add_module(
                    f"conv_{i}",
                    ConvBlock(
                        in_channels=output_filters[-1],
                        out_channels=filters,
                        kernel_size=kernel_size,
                        stride=stride,
                        pad=pad,
                        groups=groups,
                        batch_normalize=bn,
                        activation=activation
                    )
                )
            
            elif module_def['type'] == 'maxpool':
                kernel_size = int(module_def['size'])
                stride = int(module_def.get('stride',1))
                filters = output_filters[-1]
                modules.add_module(
                    f"maxpool_{i}",
                    MaxPoolBlock(
                        kernel_size=kernel_size,
                        stride=stride
                    )
                )
            
            elif module_def['type'] == 'upsample':
                stride = int(module_def.get('stride',1))
                filters = output_filters[-1]
                modules.add_module(f"upsample_{i}", UpsampleBlock(stride))
            
            elif module_def['type'] == 'route':
                layers = [int(x) for x in module_def['layers'].split(',')]
                filters = sum([output_filters[1:][l] for l in layers])
                modules.add_module(f"route_{i}",RouteBlock())
            
            elif module_def['type'] == 'shortcut':
                from_layer = int(module_def['from'])
                activation = module_def['activation']
                filters = output_filters[-1]
                modules.add_module(f"shortcut_{i}",ShortcutBlock(from_layer,activation))
            
            elif module_def['type'] == 'yolo':
                mask = [int(x) for x in module_def['mask'].split(',')]
                anchors = [int(x) for x in module_def['anchors'].split(',')]
                num_classes = int(module_def['classes'])
                scale_x_y = float(module_def.get('scale_x_y', 1.0))
                filters = output_filters[-1]
                modules.add_module(
                    f'yolo_{i}',
                    YoloBlock(
                        anchors=anchors,
                        mask=mask,
                        num_classes=num_classes,
                        scale_x_y=scale_x_y
                    )
                )

            module_list.append(modules)
            output_filters.append(filters)
        return hyperparams, module_list
    
    def forward(self,x):
        layer_outputs = []
        yolo_outputs = []
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def['type'] in ['convolutional', 'upsample', 'maxpool']:
                x = module(x)
            
            elif module_def['type'] == 'route':
                layer_indices = [int(l) for l in module_def["layers"].split(",")]
                # handle negative (like -4) and positive indices (like 47)
                layer_indices = [l if l > 0 else i + l for l in layer_indices]

                if len(layer_indices) == 1:
                    x = layer_outputs[layer_indices[0]]
                else:
                    x = torch.cat([layer_outputs[l] for l in layer_indices], dim = 1)
            
            elif module_def['type'] == 'shortcut':
                layer_i = int(module_def["from"])

                x = module[0](x, layer_outputs[i+layer_i])

            elif module_def['type'] == 'yolo':
                x = module[0](x)
                yolo_outputs.append(x)

            layer_outputs.append(x)

        return yolo_outputs if self.training else torch.cat(yolo_outputs,1)
    
# class YOLOv3(nn.Module):
#     def __init__(self, in_channels=3, num_classes=80):  
#         super().__init__()
#         self.num_classes = num_classes
#         self.in_channels = in_channels
#         self.layers = self._create_conv_layers()

#     def forward(self, x):
#         outputs = []  # for each scale
#         route_connections = []
#         for layer in self.layers:
#             if isinstance(layer, ScalePrediction):
#                 outputs.append(layer(x))
#                 continue

#             x = layer(x)

#             if isinstance(layer, ResidualBlock) and layer.num_repeats == 8:
#                 route_connections.append(x)

#             elif isinstance(layer, nn.Upsample):
#                 x = torch.cat([x, route_connections[-1]], dim=1)
#                 route_connections.pop()

#         return outputs

#     def _create_conv_layers(self):
#         layers = nn.ModuleList()
#         in_channels = self.in_channels

#         for module in config:
#             if isinstance(module, tuple):
#                 out_channels, kernel_size, stride = module
#                 layers.append(
#                     CNNBlock(
#                         in_channels,
#                         out_channels,
#                         kernel_size=kernel_size,
#                         stride=stride,
#                         padding=1 if kernel_size == 3 else 0,
#                     )
#                 )
#                 in_channels = out_channels

#             elif isinstance(module, list):
#                 num_repeats = module[1]
#                 layers.append(ResidualBlock(in_channels, num_repeats=num_repeats,))

#             elif isinstance(module, str):
#                 if module == "S":
#                     layers += [
#                         ResidualBlock(in_channels, use_residual=False, num_repeats=1),
#                         CNNBlock(in_channels, in_channels // 2, kernel_size=1),
#                         ScalePrediction(in_channels // 2, num_classes=self.num_classes),
#                     ]
#                     in_channels = in_channels // 2

#                 elif module == "U":
#                     layers.append(nn.Upsample(scale_factor=2),)
#                     in_channels = in_channels * 3

#         return layers


if __name__ == "__main__":
    IMAGE_SIZE = config.IMAGE_SIZE
    model = MyMOLO(config_path="MOLOv2v3coco.cfg")
    x = torch.randn((2, 3, IMAGE_SIZE, IMAGE_SIZE))
    out = model(x)
    assert model(x)[0].shape == (2, 3, IMAGE_SIZE//32, IMAGE_SIZE//32, config.NUM_CLASSES + 5)
    assert model(x)[1].shape == (2, 3, IMAGE_SIZE//16, IMAGE_SIZE//16, config.NUM_CLASSES + 5)
    print("Success!")

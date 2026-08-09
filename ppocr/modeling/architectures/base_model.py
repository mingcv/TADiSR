import torch.nn as nn

from ppocr.modeling.backbones import build_backbone
from ppocr.modeling.heads import build_head
from ppocr.modeling.necks import build_neck
from ppocr.modeling.transforms import build_transform


class BaseModel(nn.Module):
    def __init__(self, config, **kwargs):
        """
        the module for OCR.
        args:
            config (dict): the super parameters for module.
        """
        super(BaseModel, self).__init__()

        in_channels = config.get('in_channels', 3)
        model_type = config['model_type']
        print("Building Model named {}".format(config["model_type"]))

        if 'Transform' not in config or config['Transform'] is None:
            self.use_transform = False
        else:
            self.use_transform = True
            config['Transform']['in_channels'] = in_channels
            self.transform = build_transform(config['Transform'])
            in_channels = self.transform.out_channels
            # raise NotImplementedError
            print("Building Transform named {}".format(config["Transform"]))

        # build backbone, backbone is need for del, rec and cls
        if 'Backbone' not in config or config['Backbone'] is None:
            self.use_backbone = False
        else:
            self.use_backbone = True
            config["Backbone"]['in_channels'] = in_channels
            self.backbone = build_backbone(config["Backbone"], model_type)
            in_channels = self.backbone.out_channels
            print("Building Backbone named {}".format(config["Backbone"]))

        # build neck
        # for rec, neck can be cnn,rnn or reshape(None)
        # for det, neck can be FPN, BIFPN and so on.
        # for cls, neck should be none
        if 'Neck' not in config or config['Neck'] is None:
            self.use_neck = False
        else:
            self.use_neck = True
            config['Neck']['in_channels'] = in_channels
            self.neck = build_neck(config['Neck'])
            in_channels = self.neck.out_channels
            print("Building Neck named {}".format(config["Neck"]))

        # # build head, head is need for det, rec and cls
        if 'Head' not in config or config['Head'] is None:
            self.use_head = False
        else:
            self.use_head = True
            config["Head"]['in_channels'] = in_channels
            self.head = build_head(config["Head"], **kwargs)

            print("Building Head named {}".format(config["Head"]))
        self.return_all_feats = config.get("return_all_feats", False)

        self._initialize_weights()

    def _initialize_weights(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # print("input_size: ")
        # if isinstance(x, list):
        #     for xx in x:
        #         print(xx.shape)
        # else:
        #     print(x.shape)

        # print(
        #     f"use_transform:{self.use_transform}, use_backbone: {self.use_backbone}, use_neck: {self.use_neck}, use_head: {self.use_head}")

        y = dict()
        if self.use_transform:
            x = self.transform(x)
        if self.use_backbone:
            x = self.backbone(x)
        if isinstance(x, dict):
            y.update(x)
        else:
            y["backbone_out"] = x
        final_name = "backbone_out"

        # print("After backbone: ")
        # if isinstance(x, list):
        #     for xx in x:
        #         print(xx.shape)
        # else:
        #     print(x.shape)

        if self.use_neck:
            x = self.neck(x)
            if isinstance(x, dict):
                y.update(x)
            else:
                y["neck_out"] = x
            final_name = "neck_out"

        # print("After neck: ", x.shape)
        if self.use_head:
            x = self.head(x)

        # print("After head: ")
        # if isinstance(x, dict):
        #     for k, v in x.items():
        #         print(k, v.shape)
        # else:
        #     print(x.shape)
        # for multi head, save ctc neck out for udml
        if isinstance(x, dict) and 'ctc_nect' in x.keys():
            y['neck_out'] = x['ctc_neck']
            y['head_out'] = x
        elif isinstance(x, dict):
            y.update(x)
        else:
            y["head_out"] = x
        if self.return_all_feats:
            if self.training:
                return y
            elif isinstance(x, dict):
                return x
            else:
                return {final_name: x}
        else:
            return x

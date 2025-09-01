import copy
import torch
from torch import nn
from backbone.linears import CosineLinear, EaseCosineLinear
import timm
from tools import builder

def get_backbone(args, pretrained=False, modelconfig = None):

    name = args["backbone_type"].lower()
    if '_ease' in name:
        ffn_num = args["ffn_num"]
        if args["model_name"] == "ga" :
            # from backbone_3DShape import Point_BERT
            from easydict import EasyDict
            tuning_config = EasyDict(
                # AdaptFormer
                ffn_adapt=True,
                ffn_option= "parallel", # "sequential",
                ffn_adapter_layernorm_option="none",
                ffn_adapter_init_option="lora",
                ffn_adapter_scalar="0.1",
                ffn_num=ffn_num,
                d_model= 768,
                # VPT related
                vpt_on=False,
                vpt_num=0,
                _device = args["device"][0]
            )
            if name == "pointbert_ease":
                modelconfig.update(tuning_config)
                model = builder.model_builder(modelconfig.model)
                model.load_model_from_ckpt('pretrained_bases/mae_base.pth')
                
                for name, param in model.named_parameters():        
                    if 'point_prompt' in name or 'shift_net' in name or 'shape_feature_mlp' in name or 'adapter' in name  or 'cls_pos' in name or 'cls_token' in name or 'cls_head_' in name or "prompt_embeddings" in name or 'prompt_cor' in name or 'out_transform' in name: # or 'proj.bias' in name or 'fc2.bias' in name or 'fc1.bias' in name or 'norm2.bias' in name or 'norm1.bias' in name:
                        param.requires_grad_(True)
                    else:
                        param.requires_grad_(False)
                # model = Point_BERT.point_bert_ease(modelconfig, args)
                model.out_dim= 768 #384 
                return model
    elif '_l2p' in name:
        if args["model_name"] == "l2p":
            # from backbone import vit_l2p
            # model = timm.create_model(
            #     args["backbone_type"],
            #     pretrained=args["pretrained"],
            #     num_classes=args["nb_classes"],
            #     drop_rate=args["drop"],
            #     drop_path_rate=args["drop_path"],
            #     drop_block_rate=None,
            #     prompt_length=args["length"],
            #     embedding_key=args["embedding_key"],
            #     prompt_init=args["prompt_key_init"],
            #     prompt_pool=args["prompt_pool"],
            #     prompt_key=args["prompt_key"],
            #     pool_size=args["size"],
            #     top_k=args["top_k"],
            #     batchwise_prompt=args["batchwise_prompt"],
            #     prompt_key_init=args["prompt_key_init"],
            #     head_type=args["head_type"],
            #     use_prompt_mask=args["use_prompt_mask"],
            # )
            model = builder.model_builder(args.model)
            model.load_model_from_ckpt('pretrained_bases/mae_base.pth')
            return model

# def get_backbone_pointbert(args, pretrained=False):


class BaseNet(nn.Module):
    def __init__(self, args, pretrained, modelconfig):
        super(BaseNet, self).__init__()

        print('This is for the BaseNet initialization.')
        self.backbone = get_backbone(args, pretrained, modelconfig = modelconfig)
        print('After BaseNet initialization.')
        self.fc = None
        self._device = args["device"][0]

    @property
    def feature_dim(self):
        return self.backbone.out_dim

    def extract_vector(self, x):
        if self.model_type == 'cnn':
            self.backbone(x)['features']
        else:
            return self.backbone(x)

    def forward(self, x):
        out = self.backbone(x)
        return out

    def forward_test(self, x):
        out = self.backbone(x)
        
        return out

    def update_fc(self, nb_classes, nextperiod_initialization=None):
        fc = self.generate_fc(self.feature_dim, nb_classes).to(self._device)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            if nextperiod_initialization is not None:
                weight = torch.cat([weight, nextperiod_initialization])
            else:
                weight = torch.cat([weight, torch.zeros(nb_classes - nb_output, self.feature_dim).to(self._device)])
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc.requires_grad_(False)

    def generate_fc(self, in_dim, out_dim):
        fc = CosineLinear(in_dim, out_dim)
        return fc

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self

class EASENet(BaseNet):
    def __init__(self, args, pretrained=True, modelconfig = None):
        super().__init__(args, pretrained, modelconfig=modelconfig)
        self.args = args
        self.inc = args["increment"]
        self.last_inc = args["increment"]
        self.init_cls = args["init_cls"]
        self.nb_tasks = -1
        self._cur_task = -1
        if modelconfig is not None:
            self.out_dim = 768
        else:
            self.out_dim =  self.backbone.out_dim
        self.fc = None
            
    def freeze(self):
        for name, param in self.named_parameters():
            param.requires_grad = False
            # print(name)
    
    @property
    def feature_dim(self):
        if self.use_init_ptm:
            return self.out_dim * (self._cur_task + 2)
        else:
            return self.out_dim * (self._cur_task + 1)

    # (proxy_fc = cls * dim)
    def update_fc(self, nb_classes, inc = 0, use_exemplars = False):
        self._cur_task += 1
        if inc == 0:
            inc = self.inc
        if self._cur_task == 0:
            self.proxy_fc = self.generate_fc(self.out_dim, self.init_cls).to(self._device)
        else:
            if use_exemplars == False:
                self.proxy_fc = self.generate_fc(self.out_dim, inc).to(self._device)
            else:
                self.proxy_fc = self.generate_fc(self.out_dim, nb_classes).to(self._device)
        
        fc = self.generate_fc(self.feature_dim, nb_classes).to(self._device)
        fc.reset_parameters_to_zero()

        if self.fc is not None:
            old_nb_classes = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            fc.weight.data[ : old_nb_classes, : -self.out_dim] = nn.Parameter(weight)
        del self.fc
        self.fc = fc
    
    def generate_fc(self, in_dim, out_dim):
        fc = EaseCosineLinear(in_dim, out_dim)
        return fc
    
    def extract_vector(self, x):
        return self.backbone(x)

    def forward(self, x, test=False):
        if test == False:
            x = self.backbone.forward(x, False)
            out = self.proxy_fc(x)
        else:
            x = self.backbone.forward(x, True, use_init_ptm=self.use_init_ptm)
            if self.args["moni_adam"] or (not self.args["use_reweight"]):
                out = self.fc(x)
            else:
                if self._cur_task == self.nb_tasks - 1:
                    out = self.fc.forward_reweight(x, cur_task=self._cur_task, alpha=self.alpha, init_cls=self.init_cls, last_cls=self.last_cls, inc=self.inc, use_init_ptm=self.use_init_ptm, beta=self.beta)
                else:
                    out = self.fc.forward_reweight(x, cur_task=self._cur_task, alpha=self.alpha, init_cls=self.init_cls, inc=self.inc, use_init_ptm=self.use_init_ptm, beta=self.beta)
            # out = self.proxy_fc(x)
        out.update({"features": x})
        return out

    def show_trainable_params(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.numel())

class MOSNet(nn.Module):
    def __init__(self, args, pretrained = True, modelconfig = None):
        super(MOSNet, self).__init__()
        self.backbone = get_backbone(args, pretrained, modelconfig=modelconfig)
        if modelconfig is not None:
            self.out_dim = modelconfig.transformer_config.trans_dim #768
        else:
            self.out_dim =  self.backbone.out_dim
        self.fc = None
        self._device = args["device"][0]

    @property
    def feature_dim(self):
        return self.backbone.out_dim

    def update_fc(self, nb_classes, nextperiod_initialization=None):
        fc = self.generate_fc(self.feature_dim, nb_classes).to(self._device)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            if nextperiod_initialization is not None:
                weight = torch.cat([weight, nextperiod_initialization])
            else:
                weight = torch.cat([weight, torch.zeros(nb_classes - nb_output, self.feature_dim).to(self._device)])
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc.requires_grad_(False)
    
    def generate_fc(self, in_dim, out_dim):
        fc = CosineLinear(in_dim, out_dim)
        return fc
    
    def forward_orig(self, x):
        features = self.backbone(x, adapter_id=0, test = True)['features']
        
        res = dict()
        res['features'] = features
        res['logits'] = self.fc(features)['logits']
                
        return res
        
    def forward(self, x, adapter_id=-1, train=False, fc_only=False):
        test = not train
        res = self.backbone(x, adapter_id, test, fc_only)

        return res

# l2p and dualprompt
class PromptVitNet(nn.Module):
    def __init__(self, args, pretrained):
        super(PromptVitNet, self).__init__()
        self.backbone = get_backbone(args, pretrained)
        if args["get_original_backbone"]:
            self.original_backbone = self.get_original_backbone(args)
        else:
            self.original_backbone = None
            
    def get_original_backbone(self, args):
        model = builder.model_builder(args.original_model)
        model.load_model_from_ckpt('pretrained_bases/mae_base.pth')
        return model        
        # return timm.create_model(
        #     args["backbone_type"],
        #     pretrained=args["pretrained"],
        #     num_classes=args["nb_classes"],
        #     drop_rate=args["drop"],
        #     drop_path_rate=args["drop_path"],
        #     drop_block_rate=None,
        # ).eval()

    def forward(self, x, task_id=-1, train=False):
        with torch.no_grad():
            if self.original_backbone is not None:
                cls_features, res = self.original_backbone(x)
                # cls_features = self.original_backbone(x)['pre_logits']
            else:
                cls_features = None
        x = self.backbone(x, task_id=task_id, cls_features=cls_features, train=train)
        return x
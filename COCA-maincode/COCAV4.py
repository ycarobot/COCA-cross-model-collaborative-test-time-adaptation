import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Coca(nn.Module):
    def __init__(self, resnet, vit, optimizer,args, steps=1, episodic=False):
        super().__init__()
        self.resnet = resnet
        self.vit = vit
        self.args=args
        self.optimizer = optimizer
        self.steps = steps
        self.margin=0.4*math.log(1000)
        self.consense=False
        assert steps > 0, "COCA requires >= 1 step(s) to forward and update"
        self.episodic = episodic
        self.ids=None
        self.imagenet_mask=None

        self.T = nn.Parameter(torch.ones(1).to(self.args.gpu), requires_grad=True)
        # self.T_optimizer = torch.optim.SGD([self.T], 0.1, momentum=0.9)
        self.T_optimizer = torch.optim.AdamW([self.T], 0.01,weight_decay=0)


    def forward(self,x, x_transform):
        if self.episodic:
            self.reset()
        if self.steps > 0 and self.tryT is not True:
            for _ in range(self.steps):
                if self.args.method=='mutta_multi':
                    param_loss, final_outputs = self.forward_and_adapt(x, x_transform,self.imagenet_mask)
                    return param_loss, final_outputs

                resnet_outputs, vit_outputs, final_outputs = self.forward_and_adapt(x, x_transform,self.imagenet_mask)
                # outputs = self.forward_and_optimal_T(x, y)
        else:
            if self.tryT:
                with torch.no_grad():
                    resnet_outputs = self.resnet(x)              
                    vit_outputs = self.vit(x_transform)
                for _ in range(10):
                    T_loss = self.get_T_loss(resnet_outputs,vit_outputs)
                    T_loss.backward()
                    self.T_optimizer.step()
                    self.T_optimizer.zero_grad()

                final_outputs = (vit_outputs + resnet_outputs.detach() / self.T.detach()) / 2

        return resnet_outputs, vit_outputs, final_outputs
    
    @torch.enable_grad()
    def forward_and_adapt(self, x, x_transform,imagenet_mask):

        resnet_outputs = self.resnet(x)            
        vit_outputs = self.vit(x_transform)  

        if imagenet_mask is not None:
            resnet_outputs = resnet_outputs[:, imagenet_mask]
            vit_outputs = vit_outputs[:, imagenet_mask]
            
        for i in range(10):
            T_loss = self.get_T_loss(resnet_outputs,vit_outputs)
            T_loss.backward()
            self.T_optimizer.step()
            self.T_optimizer.zero_grad()

        if self.args.plusmethod=='sar':
            param_loss, final_outputs = self.get_simple_param_loss_sar(resnet_outputs,vit_outputs)
            param_loss.backward()
            self.optimizer.first_step(zero_grad=True)
            resnet_outputs1 = self.resnet(x)            
            vit_outputs2 = self.vit(x_transform) 
            if imagenet_mask is not None:
                resnet_outputs1 = resnet_outputs1[:, imagenet_mask]
                vit_outputs2 = vit_outputs2[:, imagenet_mask]
            param_loss_second, _ = self.get_simple_param_loss_sar(resnet_outputs1,vit_outputs2)
            param_loss_second.backward()
            self.optimizer.second_step(zero_grad=True)
            return resnet_outputs, vit_outputs, final_outputs
        
        elif self.args.plusmethod=='eata':
            param_loss, final_outputs = self.get_simple_parameata_loss(resnet_outputs,vit_outputs)
            param_loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            return resnet_outputs, vit_outputs, final_outputs
        
        if self.T.item()<50:
            param_loss, final_outputs = self.get_simple_param_loss(resnet_outputs,vit_outputs)         
        else:
            param_loss=self.get_tent_loss(vit_outputs,self.margin)
            final_outputs=vit_outputs
        

        param_loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return resnet_outputs, vit_outputs, final_outputs
    
    
    def get_simple_param_loss(self, resnet_outputs,vit_outputs):
        loss = 0       
        #SELF ENTROPY
        losseta1 = self.get_tent_loss(resnet_outputs,self.margin)
        if losseta1!=0:
            loss+=losseta1 
        losseta = self.get_tent_loss(vit_outputs,self.margin)
        if losseta!=0:
            loss+=losseta   
        final_outputs = (vit_outputs + resnet_outputs / self.T.detach()) / 2
        final_T = (final_outputs.amax(dim=-1, keepdim=True) / vit_outputs.amax(dim=-1, keepdim=True)).mean().detach()
        final_outputs = final_outputs / final_T
        if self.halfcombine:
            final_outputs=(vit_outputs + resnet_outputs) / 2

        # MARGINAL ENTROPY
        loss += 2*final_T * self.get_tent_loss(final_outputs,self.margin)
               
        #CONSENSUS LOSS
        loss +=  2*final_T*self.compute_cetent_loss(resnet_outputs,vit_outputs,final_outputs)

        return loss.mean(), final_outputs
    
    def get_simple_param_loss_sar(self, resnet_outputs,vit_outputs):
        loss = 0       
        #SELF ENTROPY
        losssar = self.get_sar_loss(resnet_outputs,self.margin)
        if losssar!=0:
            loss+=losssar 
        losssar1 = self.get_sar_loss(vit_outputs,self.margin)
        if losssar1!=0:
            loss+=losssar1 
        final_outputs = (vit_outputs + resnet_outputs / self.T.detach()) / 2

        final_T = (final_outputs.amax(dim=-1, keepdim=True) / vit_outputs.amax(dim=-1, keepdim=True)).mean().detach()
        final_outputs = final_outputs / final_T
        # MARGINAL ENTROPY
        loss += 2*final_T * self.get_sar_loss(final_outputs,self.margin)
        
        #CONSENSUS LOSS
        loss +=  2*final_T*self.compute_cesar_loss(resnet_outputs,vit_outputs,final_outputs)

        return loss.mean(), final_outputs

    def get_simple_parameata_loss(self, resnet_outputs,vit_outputs):
        loss = 0       
        #SELF ENTROPY
        losseta1 = self.get_eta_loss(resnet_outputs,self.margin)
        if losseta1!=0:
            loss+=losseta1 
        losseta = self.get_eta_loss(vit_outputs,self.margin)
        if losseta!=0:
            loss+=losseta          
        final_outputs = (vit_outputs + resnet_outputs / self.T.detach()) / 2
        final_T = (final_outputs.amax(dim=-1, keepdim=True) / vit_outputs.amax(dim=-1, keepdim=True)).mean().detach()
        final_outputs = final_outputs / final_T

        if self.halfcombine:
            final_outputs=(vit_outputs + resnet_outputs) / 2

        # MARGINAL ENTROPY
        loss += 2*final_T * self.get_eta_loss(final_outputs,self.margin)
               
        #CONSENSUS LOSS
        loss +=  2*final_T*self.compute_ceeta_loss(resnet_outputs,vit_outputs,final_outputs)

        return loss.mean(), final_outputs
    
    def compute_ceeta_loss(self, resnet_output,vit_output,combine_outptus):
        entropys = softmax_entropy(combine_outptus)

        ids = torch.where(entropys < self.margin)[0]
        if len(ids)!=0:
            self.ids=ids
        else:
            self.ids=None 
        entropys = entropys[ids]
        coeff = 1 / (torch.exp(torch.clamp(entropys.clone().detach() - self.margin, max=0)))
        criterion=nn.CrossEntropyLoss(label_smoothing=0.2,reduction='none')
        combine_probabilities = F.softmax(combine_outptus, dim=1)
        combine_preds = combine_probabilities.argmax(dim=1)
        ce_loss_resnet = criterion(resnet_output[ids], combine_preds[ids])
        ce_loss_resnet=ce_loss_resnet.mul(coeff)
        ce_loss_vit = criterion(vit_output[ids], combine_preds[ids])
        ce_loss_vit=ce_loss_vit.mul(coeff)
        return ce_loss_resnet.mean(0)+ce_loss_vit.mean(0)
    

    def compute_cetent_loss(self, resnet_output,vit_output,combine_outptus):
        entropys = softmax_entropy(combine_outptus)
        ids = torch.where(entropys < self.margin)[0]
        if len(ids)!=0:
            self.ids=ids
        else:
            self.ids=None 
            return 0
        entropys = entropys[ids]

        criterion=nn.CrossEntropyLoss(label_smoothing=0.2)
        combine_probabilities = F.softmax(combine_outptus, dim=1)
        combine_preds = combine_probabilities.argmax(dim=1)
        ce_loss_resnet = criterion(resnet_output, combine_preds)
        ce_loss_vit = criterion(vit_output, combine_preds)
        return ce_loss_resnet.mean(0)+ce_loss_vit.mean(0)
    
    def compute_cesar_loss(self, resnet_output,vit_output,combine_outptus):
        entropys = softmax_entropy(combine_outptus)
        ids = torch.where(entropys < self.margin)[0]
        if len(ids)!=0:
            self.ids=ids
        else:
            self.ids=None 
            return 0
        entropys = entropys[ids]
        criterion=nn.CrossEntropyLoss(label_smoothing=0.2)
        combine_probabilities = F.softmax(combine_outptus, dim=1)
        combine_preds = combine_probabilities.argmax(dim=1)
        ce_loss_resnet = criterion(resnet_output[ids], combine_preds[ids])
        ce_loss_vit = criterion(vit_output[ids], combine_preds[ids])

        return ce_loss_resnet.mean(0)+ce_loss_vit.mean(0)
    
    def get_T_loss(self, resnet_outputs,vit_outputs):
        anchor_output1 = vit_outputs.detach()

        anchor_output = torch.exp(anchor_output1)
        
        # loss = 0
        current_outputs1 = resnet_outputs.detach() / self.T
        current_outputs = torch.exp(current_outputs1)

        loss1= F.l1_loss(current_outputs, anchor_output)
        # loss2= F.l1_loss(current_outputs1, anchor_output1)
        loss=loss1
        return loss
    
    def get_eta_loss(self, outputs,margin):
        entropys = softmax_entropy(outputs)
        ids = torch.where(entropys < margin)[0]
        if len(ids)!=0:
            self.ids=ids
        else:
            self.ids=None 
        entropys = entropys[ids]
        coeff = 1 / (torch.exp(entropys.clone().detach() - margin))

        if len(ids)==0:
            entropys=0
        else:
            entropys = entropys.mul(coeff).mean(0)

        return entropys
    
    def get_tent_loss(self, outputs,margin):
        entropys = softmax_entropy(outputs)
        return entropys.mean(0)
    
    def get_sar_loss(self, outputs,margin):
        entropys = softmax_entropy(outputs)
        ids = torch.where(entropys < margin)[0]
        if len(ids)!=0:
            self.ids=ids
        else:
            self.ids=None 
        entropys = entropys[ids]

        return entropys.mean(0)

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    temprature = 1
    x = x/ temprature
    x = -(x.softmax(1) * x.log_softmax(1)).sum(1)
    return x

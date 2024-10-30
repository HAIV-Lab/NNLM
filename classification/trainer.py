from build import build_dataset
from model import MyModel
import torch
from fairseq import optim
import os

class Trainer(object):
    def __init__(self, args) -> None:
        self.args = args
        # 去掉了half
        print(" Start building model !!! ")
        self.model = MyModel(args).cuda()
        self.model = self.model.half()
        
        print("finish building model !!!")
    
    def train(self):
        dataloader = build_dataset(self.args)
        self.optimizer = torch.optim.SGD(self.model.parameters(), self.args.lr, 
                                         momentum=0.9, nesterov=True,
                                         weight_decay=0.0004)
        # self.optimizer = optim.FP16Optimizer.build_optimizer(self.args, params)
        self.ce = torch.nn.CrossEntropyLoss().cuda()
        scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.args.milestones, gamma=self.args.gamma)
        best_checkpoint = {
            'best_acc':0.0,
            'best_model':None
        }

        # file_path = os.path.join(self.args.save_path, 'best_checkpoint.pth')
        # checkpoint = torch.load(file_path)
        # self.model.load_state_dict(checkpoint['best_model'])


        for epoch in range(self.args.epochs):
            correct = 0
            data_len = 0
            for i, (x1, x2, label) in enumerate(dataloader):
                x1 = x1.cuda()
                x2 = x2.cuda()
                label = label.cuda()

                logits, loss, class_loss = self.model(x1, x2, label)
                
                self.optimizer.zero_grad()
              
                loss.backward()
                self.optimizer.step()
                    
                predictions = logits.argmax(dim=-1, keepdim=True)
                # correct += predictions.eq(label.view_as(predictions)).sum().item()
                
                if i % 50 == 0:
                    batch_correct = predictions.eq(label.view_as(predictions)).sum().item()
                    acc = batch_correct / x1.shape[0]

                    print(f"Train epoch: {epoch} loss: {loss}  class_loss: {class_loss} acc: {acc}")
                    if best_checkpoint['best_acc'] < acc:
                        best_checkpoint['best_acc'] = acc
                        best_checkpoint['best_model'] = self.model.state_dict()
            scheduler.step()

            if self.args.save_path is not None:
                if os.path.exists(self.args.save_path) is False:
                    os.mkdir(self.args.save_path)
                file_path = os.path.join(self.args.save_path, 'best_checkpoint.pth')
                torch.save(best_checkpoint, file_path)



                # print("update before")
                # for name, params in self.model.named_parameters():
                #     print("-->name:", name)
                #     print("-->grad_requirs:", params.requires_grad)
                #     if name.find('weight') != -1:
                #         print("-->param:", params[0][:3])
                #         if params.grad is not None:
                #             print("-->grad_value:", params.grad[0][:3])
                #     else:
                #         print("-->param:", params[:3])
                #         if params.grad is not None:
                #             print("-->grad_value:", params.grad[:3])

                # print("update after")
                # for name, params in self.model.named_parameters():
                #     print("-->name:", name)
                #     print("-->grad_requirs:", params.requires_grad)
                #     if name.find('weight') != -1:
                #         print("-->param:", params[0][:3])
                #         if params.grad is not None:
                #             print("-->grad_value:", params.grad[0][:3])
                #     else:
                #         print("-->param:", params[:3])
                #         if params.grad is not None:
                #             print("-->grad_value:", params.grad[:3])
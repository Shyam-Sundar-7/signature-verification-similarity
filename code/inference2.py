from code.model2 import LightningModel2,SiameseNetwork2
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F
class Inference_prediction2():
    def __init__(self, model_path):
        super().__init__()
        self.model_path = model_path
        s=SiameseNetwork2()
        self.model= LightningModel2.load_from_checkpoint(self.model_path,map_location="cpu",model=s)
        self.model.eval()
        self.model.freeze()
    
    def image(self,x):
        image1 = Image.open(x).convert("L")
        transformation = transforms.Compose([transforms.Resize((100,100)),
                                     transforms.ToTensor()
                                    ])
        image1 = transformation(image1)
        return image1.unsqueeze(0)

    def image1(self,x):
        image1 = x.convert("L")
        transformation = transforms.Compose([transforms.Resize((100,100)),
                                     transforms.ToTensor()
                                    ])
        image1 = transformation(image1)
        return image1.unsqueeze(0)

    def predict(self,input1,input2):
        if isinstance(input1, str):
            output,sim=self.model.forward(self.image(input1),self.image(input2))
        else:
            output,sim=self.model.forward(self.image1(input1),self.image1(input2))
        return F.pairwise_distance(output, sim).item()*100

if __name__ == "__main__":
    inference = Inference_prediction2("contro_model.ckpt")
    print(inference.predict("CEDAR/20/forgeries_20_1.png","CEDAR/11/original_11_3.png"))
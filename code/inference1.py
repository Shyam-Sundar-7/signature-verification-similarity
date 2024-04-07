from model1 import LightningModel1,SiameseNetwork1
from PIL import Image
import torchvision.transforms as transforms
import torch
class Inference_prediction1():
    def __init__(self, model_path):
        super().__init__()
        self.model_path = model_path
        s=SiameseNetwork1()
        self.model= LightningModel1.load_from_checkpoint(self.model_path,map_location="cpu",model=s)
        self.model.eval()
        self.model.freeze()
    
    def image(self,x):
        image1 = Image.open(x).convert("RGB")
        transform=transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Lambda(lambda img: img / 255.0)
        ])
        image1 = transform(image1)
        image_like1 = torch.ones_like(image1)
        return (image_like1-image1).unsqueeze(0)
    
    def image1(self,x):
        image1 = x.convert("RGB")
        transform=transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Lambda(lambda img: img / 255.0)
        ])
        image1 = transform(image1)
        image_like1 = torch.ones_like(image1)
        return (image_like1-image1).unsqueeze(0)


    def predict(self,input1,input2):
        if isinstance(input1, str):
            output,_=self.model.forward(self.image(input1),self.image(input2))
        else:
            output,_=self.model.forward(self.image1(input1),self.image1(input2))

        return output.item()*100

if __name__ == "__main__":
    inference = Inference_prediction1("models/best_model1.ckpt")
    
    print(inference.predict("CEDAR/11/original_11_1.png","CEDAR/11/original_11_3.png"))
import cv2, sys, torch, numpy as np
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh
from torchvision import transforms
from PIL import Image
from pathlib import Path

from classifier import Net

sys.path.insert(0, str(Path(__file__).resolve().parents[0] / 'models'))
from fine_tune_model import denseNet201_model

class Detect_Opt():
    def __init__(self, mode):
        self.mode = mode
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.img_size = 640
        self.weight_dir = Path('./weights')

        self.detector_name = 'yolov5s_1c.pt'
        self.classifier_name = 'cnn_sd.pt'

        if mode == 'half':
            self.detector_name = 'yolov5s_half.pt'
            self.classifier_name = 'denseNet201.pth.tar'

        self.weight_path = self.weight_dir / self.detector_name
        self.classifier_path = self.weight_dir / self.classifier_name

class App():
    def __init__(self, mode):
        opt = Detect_Opt(mode)
        self.mode = mode

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.half = device.type != 'cpu'
        self.device = device
        print('device:', device)

        model = attempt_load(opt.weight_path.as_posix(), map_location=device)
        self.imgsz = check_img_size(opt.img_size, s=model.stride.max())
        if self.half:
            model.half()  # to FP16
        self.model = model
        self.opt = opt

        self.transform = transforms.Compose([
            transforms.Resize((32,32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])
        ])
        if self.mode == 'head':
            self.classifier = Net()
            map_location = torch.device('cpu')
        else:
            self.classifier, _ = denseNet161_model()
            map_location = lambda storage, loc: storage
        self.classifier.load_state_dict(torch.load(
            opt.classifier_path.as_posix(),
            map_location=map_location
        ))
        self.classifier.eval()
    
    def cut_and_pred(self, img0, boxes):
        heads = []
        for xyxy in boxes:
            x0, y0, x1, y1 = map(int, xyxy)
            head = Image.fromarray(img0[y0:y1, x0:x1])
            head = self.transform(head)
            heads.append(head)
        pred = self.classifier(torch.stack(heads))
        return torch.max(pred,1)[1]

    
    def detect_img(self, img0, txtpath=None):
        img = letterbox(img0, new_shape=self.imgsz)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # print(img.shape)
        pred = self.model(img)[0]
        pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, agnostic=True)
        self.gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]
        if txtpath:
            det = pred[0]
            if not len(det): return
            det[:,:4] = scale_coords(img.shape[2:], det[:,:4], img0.shape).round()
            
            boxes = det[:,:4]
            classes = self.cut_and_pred(img0, boxes)
            det[:, -1] = classes

            with open(txtpath, 'w') as f:
                f.writelines(map(self.change, det))
            pred[0] = det
        return pred
    
    def change(self, record):
        *xyxy, conf, c = record
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / self.gn).view(-1).tolist()
        return ('%d' + ' %f' * 4 + '\n') % (int(c), *xywh)


if __name__ == "__main__":
    app = App('half')
    img = cv2.imread('images/class1/a.jpg')
    pred = app.detect_img(img, 'a.txt')
    print(len(pred))
    print(pred)
    

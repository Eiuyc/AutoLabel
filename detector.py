import cv2, os, torch, numpy as np
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh
from classifier import Net
from torchvision import transforms
from PIL import Image

class Detect_Opt():
    def __init__(self):
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.weight_path = 'weights/yolov5s_1c.pt'
        self.img_size = 640
        self.classifier_path = './weights/CNN.pt'

class App():
    def __init__(self):
        opt = Detect_Opt()
        # device = 'cpu' or '0' or '0,1,2,3'
        device = 'cpu'

        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        cpu = device == 'cpu'
        if not cpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = device
            assert torch.cuda.is_available(), f'CUDA unavailable, invalid device {device} requested'

        cuda = not cpu and torch.cuda.is_available()
        device = torch.device('cuda:0' if cuda else 'cpu')
        self.half = device.type != 'cpu'
        self.device = device
        print('device:', device)

        model = attempt_load(opt.weight_path, map_location=device)
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
        self.classifier = Net()
        self.classifier.load_state_dict(
            torch.load(opt.classifier_path, map_location=torch.device('cpu'))
        )
        # self.classifier = self.classifier.cpu()
        self.classifier.eval()
        # self.classifier.cuda()
    
    def cut_and_pred(self, img0, boxes):
        heads = []
        for xyxy in boxes:
            x0, y0, x1, y1 = map(int, xyxy)
            head = Image.fromarray(img0[y0:y1, x0:x1])
            head = self.transform(head)
            heads.append(head)
        pred = self.classfier(torch.stack(heads))
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
            
            # boxes = det[:,:4]
            # classes = self.cut_and_pred(img0, boxes)
            # det[:, -1] = classes

            with open(txtpath, 'w') as f:
                f.writelines(map(self.change, det))
            pred[0] = det
        return pred
    
    def change(self, record):
        *xyxy, conf, c = record
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / self.gn).view(-1).tolist()
        return ('%d' + ' %f' * 4 + '\n') % (int(c), *xywh)


if __name__ == "__main__":
    app = App()
    img = cv2.imread('images/test.jpg')
    pred = app.detect_img(img, 'test__.txt')
    print(len(pred))
    print(pred)
    


# video_path = './videos/exam.mkv'
# video_name = video_path.split('/')[-1].split('.')[0]
# img_dir = './images/'+ video_name

# if not os.path.exists(img_dir):
#     os.mkdir(img_dir)

# cap = cv2.VideoCapture(video_path)
# i = 0
# while 1:
#     _, frame = cap.read()
#     i+=1
#     if i%30 == 0:
#         cv2.imwrite(f'{img_dir}/{video_name}_{i:05}.jpg', frame)
#     # cv2.imwrite(f'{img_dir}/{video_name}_{i:05}.jpg', frame)




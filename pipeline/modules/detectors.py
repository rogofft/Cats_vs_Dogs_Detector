from pipeline.modules.resnet import ResNet18
import torch
import torch.nn as nn

from torchvision.models import inception_v3, resnet50
from pipeline.modules.utils import get_best_anchor
from pipeline import device
import torchvision.transforms as T


# User's models

# #########################################
class AbstractModel(nn.Module):
    """
    Base Abstract class for detector models.
    """
    def forward(self, x):
        x = self.backbone(x)
        x = self.out_layer(x)
        # apply activation function, corresponding to the output neuron:
        # [0:sigmoid, 1:sigmoid, 2:exp, 3:exp, 4+:softmax]
        x_reg, x_size, x_class = torch.sigmoid(x[:, :2]), torch.exp(x[:, 2:4]), x[:, 4:]
        bbox = torch.cat([x_reg, x_size], dim=1)
        return bbox, x_class

    def predict(self, imgs: tuple):
        # prepare images for net
        img_tensors_list = []
        img_size_list = []
        for img in imgs:
            # get x, y sizes of image
            img_size_list.append(torch.tensor(img.size, dtype=torch.float).unsqueeze_(0))
            # transform image to tensor
            img_tensors_list.append(self.transforms(img).unsqueeze_(0))

        # Concatenate all tensors to batch
        img_tensor = torch.cat(img_tensors_list, dim=0)
        size_tensor = torch.cat(img_size_list, dim=0)

        # forward
        with torch.no_grad():
            anchors, logits = self(img_tensor.to(device))

            tensor_cls = torch.argmax(torch.softmax(logits, dim=1), dim=1) + 1

            if self.predict_anchors:
                # Get bbox in relative coordinates
                bbox = get_best_anchor(anchors).cpu()
            else:
                bbox = anchors.cpu()

        # Left Up corners
            tensor_x1 = (torch.clamp(bbox[:, 0] - bbox[:, 2] / 2., min=0., max=1.) * size_tensor[:, 0]).long()
            tensor_y1 = (torch.clamp(bbox[:, 1] - bbox[:, 3] / 2., min=0., max=1.) * size_tensor[:, 1]).long()
        # Right bottom corners
            tensor_x2 = (torch.clamp(bbox[:, 0] + bbox[:, 2] / 2., min=0., max=1.) * size_tensor[:, 0]).long()
            tensor_y2 = (torch.clamp(bbox[:, 1] + bbox[:, 3] / 2., min=0., max=1.) * size_tensor[:, 1]).long()

        # Make out list
        output_list = []
        for cls, x1, y1, x2, y2 in zip(tensor_cls, tensor_x1, tensor_y1, tensor_x2, tensor_y2):
            output_list.append((cls.item(), x1.item(), y1.item(), x2.item(), y2.item()))

        return output_list


# #########################################
# Implementation of Resnet18
class BaselineModel(AbstractModel):
    """
    ResNet18-based baseline model detection/classification
    """

    def __init__(self):
        super(BaselineModel, self).__init__()
        self.predict_anchors = False
        self.backbone = ResNet18()
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        self.out_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 6, bias=True)
        )

        # Normalize and resize image for predict function
        self.transforms = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor()
        ])


# #########################################
class TransferInceptionV3(AbstractModel):
    """
    Inception_v3 transfer learning model for detection/classification
    """

    def __init__(self):
        super(TransferInceptionV3, self).__init__()
        self.predict_anchors = False
        self.backbone = inception_v3(pretrained=True)
        self.backbone.aux_logits = False

        # Hold backbone weights
        for param in self.backbone.parameters():
            param.requires_grad = False

        # New trainable out layer
        self.backbone.fc = nn.Linear(2048, 6, bias=True)

        # Normalize and resize image for predict function
        self.transforms = T.Compose([
            T.Resize((299, 299)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def forward(self, x):
        x = self.backbone(x)
        # apply activation function, corresponding to the output neuron:
        # [0:sigmoid, 1:sigmoid, 2:exp, 3:exp, 4+:softmax]
        x_reg, x_size, x_class = torch.sigmoid(x[:, :2]), torch.exp(x[:, 2:4]), x[:, 4:]
        bbox = torch.cat([x_reg, x_size], dim=1)
        return bbox, x_class


# #########################################
# Implementation of Resnet18
class TransferResnet50(AbstractModel):
    """
    ResNet50 transfer learning model with 2 Dense layers at out
    """

    def __init__(self):
        super(TransferResnet50, self).__init__()
        self.predict_anchors = False

        self.backbone = nn.Sequential(*list(resnet50(pretrained=True).children())[:-2])
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.out_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(2048, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 6, bias=True)
        )

        # Normalize and resize image for predict function
        self.transforms = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


# #########################################
# Pretrained ResNet50 with 7x7x3 anchors
class BestDetectorEver(AbstractModel):
    def __init__(self):
        super(BestDetectorEver, self).__init__()
        self.predict_anchors = True

        self.backbone = nn.Sequential(*list(resnet50(pretrained=True).children())[:-2])
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(2048, 2, bias=True)
        )

        self.bbox_predictor = nn.Sequential(
            nn.Conv2d(2048, 15, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True),
            nn.BatchNorm2d(15)
        )

        # Shapes of anchors
        self.anchor_shape_A = (0.5, 2.)
        self.anchor_shape_B = (1., 1.)
        self.anchor_shape_C = (2., 0.5)

        # Normalize and resize image for predict function
        self.transforms = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def forward(self, x):
        features = self.backbone(x)

        logits = self.classifier(features)
        bboxes = self.bbox_predictor(features)

        # AnchorA
        pa = torch.sigmoid(bboxes[:, :1])
        xa = torch.sigmoid(bboxes[:, 1:2])
        ya = torch.sigmoid(bboxes[:, 2:3])
        wa = torch.exp(bboxes[:, 3:4]) * self.anchor_shape_A[0]
        ha = torch.exp(bboxes[:, 4:5]) * self.anchor_shape_A[1]

        # AnchorB
        pb = torch.sigmoid(bboxes[:, 5:6])
        xb = torch.sigmoid(bboxes[:, 6:7])
        yb = torch.sigmoid(bboxes[:, 7:8])
        wb = torch.exp(bboxes[:, 8:9]) * self.anchor_shape_B[0]
        hb = torch.exp(bboxes[:, 9:10]) * self.anchor_shape_B[1]

        # AnchorC
        pc = torch.sigmoid(bboxes[:, 10:11])
        xc = torch.sigmoid(bboxes[:, 11:12])
        yc = torch.sigmoid(bboxes[:, 12:13])
        wc = torch.exp(bboxes[:, 13:14]) * self.anchor_shape_C[0]
        hc = torch.exp(bboxes[:, 14:15]) * self.anchor_shape_C[1]

        anchors = torch.cat([pa, xa, ya, wa, ha, pb, xb, yb, wb, hb, pc, xc, yc, wc, hc], dim=1)
        return anchors, logits

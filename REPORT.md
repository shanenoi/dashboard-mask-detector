# *I. Nhận Diện Đeo Khẩu Trang Và Các Hướng Giải Quyết*
## 1. Sử dụng mạng YOLOv5 (Trung Nguyên)
### * Phân Tích Vấn Đề Và Đưa Ra Giải Pháp
> Các bước cần thực hiện là phân vùng khuông mặt đồng thời phân loại có đeo khẩu trang hay không. Như vậy, input là một bức ảnh(1 frame của video) output sẽ cho ra các bouding box chưa khuôn mặt và kèm theo label classified như có đeo khẩu trang hay không. Ta sẽ sử dụng mạng YOLOv5 áp dụng cho use case này.
### * YOLOv5 Là Gì, Nó Có Ăn Được Không?
> **YOLOv5** là một mạng *YOLO* được phát triển và sử dụng với *phiên bản lần thứ 5*. *YOLO* trong lĩnh vực *Computer Vision* là một mô hình thường được dùng để trích xuất các đặt tính nổi bật và nhận diện các đối tượng (như con người, các loài vật chó mèo,...) được kết hợp bởi hai layer phổ biến là *Convolutional Neural Networks*(aka CNN) và *Full Connected Layer*. *YOLO* với v5 thì được cải thiện tốc độ và độ chính xác với việc sử dụng framework *PyTorch (của Facebook*) thuận tiện hơn trong việc trainning (chỉ với vài một dòng lệnh!).

### * Triển Khai Trainning
> - Dataset: Được lấy từ [Bài Post](https://www.kaggle.com/andrewmvd/face-mask-detection) của bác `Andrew Maranhão`
> - YOLO Model: *YOLO* có cung cấp sẵn cho chúng ta các model với các kích thước khác nhau ứng theo dung lượng và tốc độ xử lý như `S`, `M`, `L`, `X` với size `S` nhỏ nhất và nhanh nhất. Để có độ chính xác và tốc độ xử lý thì chúng em chọn size `M` (theo kinh nghiệm với hầu hết các tuto đa số là chọn size `M`).
> - Classes:
>   - without_mask 
>   - with_mask 
>   - mask_weared_incorrect 
> - Hyper-Parameter:
>   - Batch: 16
>   - Epochs: 100
>
> ======================
>
> *Thông tin YOLO Model*
```
                 from  n    params  module                                  arguments                     
  0                -1  1      5280  models.common.Conv                      [3, 48, 6, 2, 2]              
  1                -1  1     41664  models.common.Conv                      [48, 96, 3, 2]                
  2                -1  2     65280  models.common.C3                        [96, 96, 2]                   
  3                -1  1    166272  models.common.Conv                      [96, 192, 3, 2]               
  4                -1  4    444672  models.common.C3                        [192, 192, 4]                 
  5                -1  1    664320  models.common.Conv                      [192, 384, 3, 2]              
  6                -1  6   2512896  models.common.C3                        [384, 384, 6]                 
  7                -1  1   2655744  models.common.Conv                      [384, 768, 3, 2]              
  8                -1  2   4134912  models.common.C3                        [768, 768, 2]                 
  9                -1  1   1476864  models.common.SPPF                      [768, 768, 5]                 
 10                -1  1    295680  models.common.Conv                      [768, 384, 1, 1]              
 11                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']          
 12           [-1, 6]  1         0  models.common.Concat                    [1]                           
 13                -1  2   1182720  models.common.C3                        [768, 384, 2, False]          
 14                -1  1     74112  models.common.Conv                      [384, 192, 1, 1]              
 15                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']          
 16           [-1, 4]  1         0  models.common.Concat                    [1]                           
 17                -1  2    296448  models.common.C3                        [384, 192, 2, False]          
 18                -1  1    332160  models.common.Conv                      [192, 192, 3, 2]              
 19          [-1, 14]  1         0  models.common.Concat                    [1]                           
 20                -1  2   1035264  models.common.C3                        [384, 384, 2, False]          
 21                -1  1   1327872  models.common.Conv                      [384, 384, 3, 2]              
 22          [-1, 10]  1         0  models.common.Concat                    [1]                           
 23                -1  2   4134912  models.common.C3                        [768, 768, 2, False]          
 24      [17, 20, 23]  1     32328  models.yolo.Detect                      [3, [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]], [192, 384, 768]]
Model Summary: 369 layers, 20879400 parameters, 20879400 gradients, 48.1 GFLOPs

Transferred 475/481 items from yolov5m.pt
Scaled weight_decay = 0.0005
optimizer: SGD with parameter groups 79 weight, 82 weight (no decay), 82 bias
albumentations: version 1.0.3 required by YOLOv5, but version 0.1.12 is currently installed
train: Scanning 'mask/train/labels' images and labels...765 found, 0 missing, 0 empty, 0 corrupted: 100% 765/765 [00:05<00:00, 152.58it/s]
train: New cache created: mask/train/labels.cache
val: Scanning 'mask/test/labels' images and labels...88 found, 0 missing, 0 empty, 0 corrupted: 100% 88/88 [00:00<00:00, 95.13it/s]
val: New cache created: mask/test/labels.cache
Plotting labels to runs/train/exp/labels.jpg...
```
> *Sau khi train*
```
Model Summary: 290 layers, 20861016 parameters, 0 gradients, 48.0 GFLOPs
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 3/3 [00:05<00:00,  1.99s/it]
                 all         88        396      0.881       0.84      0.845        0.6
        without_mask         88         53      0.902      0.849      0.863      0.591
           with_mask         88        331      0.924      0.921      0.945      0.671
mask_weared_incorrect         88         12      0.818      0.749      0.726      0.539
```
> *Release [v0.0.1](https://github.com/shanenoi/dashboard-mask-detector/releases/tag/v0.0.1)*

***Kết Quả: sau các loại tối ưu siêu tham số, mô hình có độ chính xác cao nhưng không thích hợp cho vấn đề realtime vì weight của model khá nặng, thực hiện nhiều phép tính nên có độ delay khá cao trên một frame(đâu đó tầm 2-3s) nên không phù hợp cho trường hợp này.***

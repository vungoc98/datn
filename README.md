## A. Giới thiệu
Phát hiện và nhận diện biển báo giao thông là bài toán object detection
Sử dụng mô hình SSD với các backbone VGG16, MobileNetv2 để áp dụng vào bài toán
## B. Cài đặt môi trương
- Cài đặt hệ điều hành Ubuntu 16.04
#### Clone project
  git clone https://github.com/vungoc98/datn.git
#### Cài đặt các thư viện
  cd /datn/datn_backup/ && pip install -r requirements.txt
## C. Download dataset
- Vào link: https://sid.erda.dk/public/archives/ff17dc924eba88d5d01a807357d6614c/published-archive.html
- Download file zip: FullIJCNN2013.zip
- unzip file FullIJCNN2013.zip
## D. Training project trên Google Colab
#### Bước 1:
  Zip foder datn_backup vừa clone. Sau đó tải các file zip datn_backup và datasets lên drive
#### Bước 2: 
  Vào Google Colab tạo new notebook: File -> New notebook
#### Bước 3: 
  Chạy lần lượt các lệnh sau
  - %tensorflow_version 1.x
  - from google.colab import drive
    drive.mount('/content/gdrive')
  - !mkdir /train
  - !cp /content/gdrive/My\ Drive/datn_backup.zip /train
    !cd /train && unzip datn_backup.zip
  - !cd /train/datn_backup && mkdir datasets 
    !cp /content/gdrive/My\ Drive/FullIJCNN2013.zip /train/datn_backup/datasets
    !cd /train/datn_backup/datasets && unzip FullIJCNN2013.zip
  - !pip3 install keras==2.2.4
 Training MobileNetv2 + SSD512 + sử dụng splitting image trong quá trình training:
  - !cd /train/datn_backup/training && python train_mobilenetv2ssd512_last.py 
 Training không sử dụng splitting image trong quá trình training:
 Mở file datn_backup/data_generator/object_detection_2d_data_generator.py comment từ dòng 1062 đến 1108 xong chạy
  - !cd /train/datn_backup/training && python train_vgg16ssd300_last.py # VGG16 + SSD300 
  - !cd /train/datn_backup/training && python train_mobilenetv2ssd512_last.py # MobileNetv2 + SSD512 

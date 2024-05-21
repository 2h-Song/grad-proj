import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os
import shutil

# 클래스 레이블
classes = ['캔', '유리', '플라스틱']

# CNN 모델 정의 (학습 스크립트와 동일하게 유지)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 56 * 56, 512)
        self.fc2 = nn.Linear(512, 3)  # 출력 노드 개수를 3으로 설정합니다.

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 56 * 56)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 모델 초기화 및 로드
model = CNN()
model.load_state_dict(torch.load('cnn_model.pth'))
model.eval()

# 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 예측 함수 정의
def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')  # 이미지 열 때 RGB 형식으로 변환
    image = transform(image).unsqueeze(0)  # 배치 차원 추가
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        return classes[predicted[0]]

# 새로운 이미지 예측 및 추적
input_dir = './input'

# 디렉토리 내의 모든 이미지 파일 검사
image_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.jpg') or f.endswith('.png')])

# 새로운 이미지 예측 및 출력
for filename in image_files:
    image_path = os.path.join(input_dir, filename)
    prediction = predict_image(image_path)
    print(f"{filename}의 사진은 {prediction}입니다.")
    
    # 파일을 휴지통으로 이동 (윈도우 환경)
    try:
        shutil.move(image_path, os.path.join(os.getenv('USERPROFILE'), 'Desktop', '휴지통'))
    except Exception as e:
        print(f"파일 삭제 오류: {e}")

print('모든 이미지의 예측이 완료되었습니다.')

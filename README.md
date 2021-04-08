# AutoML
![wandb_logo](https://user-images.githubusercontent.com/68543150/113981579-c993a900-9882-11eb-8c88-9fcde4f8d676.png)

Wandb 라이브러리를 이용한 자동화
- 리드미에서는 간략하게 MNIST을 사용하여 wandb모듈을 어떻게 사용하고 자동화하는지 알아본다.

## How to Use?
- 먼저 [Wandb 사이트](https://wandb.ai/home)에서 회원가입을 해야한다. 추후에 연동을 통해 __모델 학습의 결과가 자동으로 본인 워크스페이스에 저장이 되기 때문__ 이다.

- 그 다음 파이썬에서 라이브러리를 임포트한다.
  - `import wandb`
- 그 다음 초기설정을 하면 키를 입력하라고 나오는데, 출력문에 나오는 링크로 들어가 키를 복사한 후 엔터를 하면 된다.
  -   `wandb.init()`

예제 코드는 [MNIST+wandb+sweep.ipynb]()에서 확인할 수 있다! 
  -   해당 코드는 colab에서 진행하였다! 설명문을 아래 첨부하였다.
## AutoML Monitoring
### 1. tensorflow 모듈 import
```python3
!pip install -q tensorflow-gpu==2.0.0-rc1
import tensorflow as tf
```
- colab은 실행할 때 마다 모듈을 설치해야 한다. `!`을 붙이면 터미널에서 실행한다는 의미이다.
- `-q` 옵션을 통해 설치할 때 나오는 옵션을 숨길 수 있다.
### 2. wandb 모듈 import
```python3
!pip install -q wandb
import wandb 
from wandb.keras import WandbCallback
```
- tensorflow와 마찬가지로 wandb 모듈을 설치하고 import 한다.
- 해당 노트북에서는 keras 모델을 사용하므로 `wandb.keras`에서 `WandbCallback`을 임포트한다. 모델이 실행될 때마다 로그가 저장된다.
### 3. wandb 초기화 및 설정
```python3
wandb.init(project='mnist-tf2')
config = wandb.config
config.learning_rate = 0.02
config.dropout_rate = 0.3
config.hidden1 = 256
config.activation1 = 'tanh'
```
- `wandb.init()`으로 프로젝트를 만들 수 있다. 본인의 wandb.ai 사이트에 프로젝트명(mnist-tf2)을 가진 워크스페이가 생성된다.
- `config`을 통해 모델 하이퍼파라미터를 변수로 지정할 수 있다. 
### 4. MNIST 데이터 불러오기
```python3
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0,  x_test / 255.0
```
- mnist 데이터를 불러와 train, test 데이터 셋으로 분할하고, 255로 나눠 픽셀 값을 0~1사이의 값으로 정규화한다.
### 5. Model Build
```python3
%%wandb
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(config.hidden1, activation=config.activation1),
    tf.keras.layers.Dropout(config.dropout_rate),
    tf.keras.layers.Dense(10, activation='softmax')
])
opt = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
model.compile(optimizer=opt,
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])
```
- 간단한 Keras Sequential 모델을 만들어봤다.
- Dense 레이어를 보면 원래 `Dense(128, activation='relu')`처럼 쓰여야하나, 위에서 선언한 wandb `config` 변수를 사용하면,
- 한번에 파라미터들을 변경할 수 있기 때문에 나중에 하이퍼파라미터를 관리하기 용이해진다. 
- `%%wandb`를 선언하면 마치 `%matplotlib inline`옵션 처럼 주피터 노트북 안에서 모델 그래프를 확인할 수 있다.
  - 그러나 여러번 반복하면 제대로 반영하지 못하므로, 반복 실행시에는 본인의 wandb.ai 워크스페이스에서 확인하는 것이 좋다. (더 interactive하게 관찰할 수 있는 장점이 있다.)


![workspace_wandb](https://user-images.githubusercontent.com/68543150/113983189-9c47fa80-9884-11eb-9caf-b7bb1af8a0ba.png)
### 6. Train & Evaluation
```python3
model.fit(x_train, y_train, 
          validation_data = (x_test, y_test), # 중간중간에 성능 체크
          epochs=5, callbacks=[WandbCallback()])

model.evaluate(x_test, y_test, verbose=2)
```
- 마지막으로 모델을 적합시킨다.
- `WandbCallback()`은 로컬 환경에서 실행하면 해당 노트북이 있는 경로에 wandb폴더 안에 모델 log가 저장되지만, colab에서는 되지 않는다. 
  - 그러나 wandb.ai 워크스페이스에도 자동 저장되니 걱정하지 말자!
### 7. Overview
![overview](https://user-images.githubusercontent.com/68543150/113984475-1af16780-9886-11eb-9276-9298f663fecd.png)
워크스페이스 overview tab에서 각 모델이 어떤 하이퍼 파라미터를 사용했는지 확인할 수 있다.


## Sweep
- 우리의 목표는 자동화이다. `config.` 옵션을 일일이 바꾸는 것으로는 만족할 수 없다ㅠ
- 이러한 니즈(?)를 만족시켜주기 위해 wandb에서는 sweep을 제공한다.
![sweep_logo](https://user-images.githubusercontent.com/68543150/113985403-14172480-9887-11eb-8066-7cdc55df896d.png)
- 위의 sweep - create sweep을 누르면 똑똑한 wandb가 config 파라미터를 반영하여 랜덤하게 설정해준다.
```python3
program: train.py
method: bayes
metric:
  goal: minimize
  name: _step
parameters:
  learning_rate:
    max: 0.08886337852238066
    min: 0.0005
    distribution: uniform
  dropout_rate:
    max: 1.1144017627651885
    min: 0.07452731489011441
    distribution: uniform
  activation1:
    values:
      - relu
      - sigmoid
      - tanh
    distribution: categorical
  hidden1:
    max: 850
    min: 47
    distribution: int_uniform
```
- create sweep을 누르면 나오는 코드. 할 때마다 다르게 나온다.
- 각 하이퍼 파라미터 범위를 설정하면 해당 범위안에서 자동으로 하이퍼 파라미터를 설정하고 학습을 진행할 수 있다. (다 쓸고 지나간다는 의미에서 sweep인가 보다.)
- sweep을 진행하기 전 두 가지 준비사항이 있다.

### 1. config을 딕셔너리 형태로 변경
```python3
wandb.init(project='mnist-tf2')
config = wandb.config
config.learning_rate = 0.02
config.dropout_rate = 0.3
config.hidden1 = 256
config.activation1 = 'tanh'
```
을
```python3
default_config = {
    'learning_rate': 0.02,
    'dropout_rate': 0.3,
    'hidden1': 256,
    'activation1': 'tanh'
}
wandb.init(project='mnist-tf2-sweep', config=default_config)
config = wandb.config
```
로 바꿔주면 된다.

### 2. train.py파일을 colab에 저장
- colab을 사용하는 경우 해당 노트북의 코드를 train.py로 합쳐 업로드 하면 된다.
- 자세한 코드는 [train.py](https://github.com/Haebuk/AutoML/blob/main/train.py)을 참고하면 된다.
- .ipynb는 실행해본 결과 호환이 되지 않는 것 같다.

준비 과정을 다 마쳤다면, 아래 코드를 실행해보자.
```python3
!wandb agent kade/mnist-tf2/rq02mc7g
```
- 여기서 rq02mc7g 대신 create sweep시 나오는 코드를 붙여넣으면 된다.
![agent](https://user-images.githubusercontent.com/68543150/113989884-d5379d80-988b-11eb-8b9d-e722b3c8080b.png)
- 멋진 에이전트와 함께 모델 학습을 무한으로 즐길 수 있다. 돌아가는 동안 월급 루팡(?)을 하면 된다.
![sweepgraphs](https://user-images.githubusercontent.com/68543150/113990530-83dbde00-988c-11eb-985f-844b5a489aa4.png)
- 잠깐 돌린 사이 벌써 15개의 하이퍼 파라미터 튜닝을 진행하였다. 
- 학습을 멈추고 싶다면, sweep 워크 스페이스에서 sweep control 탭에서 정지를 하면 된다.

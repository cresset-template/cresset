# 범용 PyTorch 소스 빌드 Docker 템플릿

Credit: [@dlwnsgud8406](https://github.com/dlwnsgud8406)

## Preamble
최근 몇 년 동안 더 작고 효율적인 장치에서 계속 증가하는 데이터 양에 대처하기 위해
효율적인 신경망을 설계하고 구현하는 데 엄청난 학문적 노력이 투입되었습니다.
그러나 이 글을 쓰는 시점에서 대부분의 딥 러닝 실무자들은 가장 기본적인 GPU 가속 기술조차 모르고 있습니다.

특히 학회에서는 메모리 요구 사항을 1/4로 줄이고,
속도를 4~5배 높일 수 있는 AMP(Automatic Mixed Precision)조차 사용하지 않는 경우가 많습니다.
[HuggingFace Accelerate](https://github.com/huggingface/accelerate) 또는 [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning)를 사용하여 큰 번거로움 없이 AMP를 활성화할 수 있음에도 마찬가지입니다.
특히 Accelerate 라이브러리는 몇 줄의 코드만으로 기존 PyTorch 프로젝트에 통합할 수 있습니다.

딥러닝의 신비에 발을 담그기 시작한 초보자라도 더 많은 컴퓨팅이 성공의 핵심 요소라는 것을 알고 있습니다.
과학자가 아무리 똑똑하더라도 10배 더 많은 컴퓨팅으로 경쟁자를 능가하는 것은 결코 대단한 일이 아닙니다.

이 템플릿은 GPU, CUDA, Docker 등에 대한 지식이 많지 않은 연구원과 엔지니어가 __*동일한 하드웨어와 신경망을 사용하여*__ GPU의 성능을 최대한 끌어낼 수 있도록 하기 위해 만들어졌습니다.

PyTorch 소스 빌드가 포함된 Docker 이미지는 이미 공식 [PyTorch Docker Hub](https://hub.docker.com/r/pytorch/pytorch)레포지토리와 [NVIDIA NGC](https://ngc.nvidia.com/catalog/containers/nvidia:pytorch) 레포지토리에서 사용할 수 있지만 이러한 이미지에는 다른 패키지가 많이 설치되어 있어 기존 프로젝트에 통합하기 어렵습니다.
또한 많은 실무자는 Docker 이미지보다 로컬 환경을 사용하는 것을 선호합니다.

여기에 제시된 프로젝트는 다릅니다.
사용자가 설치한 라이브러리를 제외하고 작업할 추가 라이브러리가 없습니다.
빌드에서 생성된 휠은 Docker 사용법을 배울 필요 없이 모든 환경에서 사용하기 위해 추출할 수 있습니다.
 (이 프로젝트의 두 번째 부분에서는 Docker를 훨씬 쉽게 사용할 수 있도록 `docker-compose.yaml` 파일도 제공합니다.)

만약 당신이 더 빨리 끝내고 싶어하는 사람이라면 Tensor board를 응시하면서 오랜 시간을 견디는것은 끝났습니다.
이 프로젝트가 바로 정답일 것입니다.
AMP와 결합된 최신 버전의 CUDA와 함께 PyTorch의 소스 빌드를 사용할 때,
순수한 PyTorch 환경보다 학습/추론시간을 10배 빠르게 달성할 수 있습니다.

제 프로젝트가 학계와 산업계의 실무자들에게 도움이 되기를 진심으로 바랍니다.
제 작업이 유익하다가 생각하시는 사용자는 이 저장소에 star을 해주셔서 감사를 표시해주시는 것을 환영합니다.


## Warning
__*이 템플릿을 사용하기 전에 먼저 GPU를 실제로 사용하고 있는지 확인하세요!*__

대부분의 시나리오에서, 느린 학습은 비효율적인 파이프라인 ETL(추출, 변환, 로드)에 의해 발생합니다.
데이터가 GPU가 느리게 실행되기 때문이 아니라 데이터가 GPU에  충분히 빠르게 도달하지 않아서 학습이 느립니다.
GPU 사용률이 컴퓨팅 최적화를 할 만큼 충분히 높은지 확인하려면 `watch nvidia-smi`를 실행하세요.
GPU 사용률이 낮거나 돌발적으로 최고조에 달하는 경우, 이 템플릿을 사용하기 전에 효율적인 ETL 파이프라인을 설계하세요.
그렇지 않으면, 더 빠른 컴퓨팅이 병목 현상이 되지 않으므로 별로 도움이 되지 않습니다.

효율적인 ETL 파이프라인 설계에 대한 가이드는 https://www.tensorflow.org/guide/data_performance 를 참조하세요.

[NVIDIA DALI](https://github.com/NVIDIA/DALI) 라이브러리 또한 도움이 될 수도 있습니다. 
The [DALI PyTorch plugin](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/plugins/pytorch_tutorials.html)
은 PyTorch에서 효율적인 ETL 파이프라인을 위한 API를 제공합니다.


## Introduction
PyTorch/CUDA/cuDNN의 __*모든 버전*__ 에서의 __*소스로부터*__ PyTorch를 빌드하기 위한 템플릿 레포지토리.

소스에서 빌드한 PyTorch는 `pip`/`conda`에서 설치된 PyTorch보다 훨씬 빠르지만(일부 벤치마크에서는 x4배, x2가 더 일반적입니다.)
소스에서 빌드하는 것은 힘들고 버그가 발생하기 쉬운 프로세스입니다.

이 레포지토리는 모든 버전의 CUDA에서의 소스로부터 모든 버전의 PyTorch를 빌드하기 위한 고도로 모듈화된 템플릿입니다.
Linux 기반 이미지 또는 프로젝트에 통합할 수 있는 사용하기 쉬운 Dockerfile을 제공합니다.

Docker에 익숙하지 않은 연구원을 위해 생성된 휠 파일을 추출하여 로컬 환경에 PyTorch를 설치할 수 있습니다.

Windows 사용자는 WSL을 통해 이 프로젝트를 사용할 수도 있습니다. 아래 지침을 참조하세요.

`Makefile`은 쉬운 사용을 위한 인터페이스와 맞춤형 이미지 구축을 위한 튜토리얼로 제공됩니다.

Docker를 사용한 간단한 대화형 개발 환경을 위해 `docker-compose.yaml` 파일도 제공됩니다.

이 템플릿의 속도 향상은 다음 요소에서 비롯됩니다.
1. 최신 버전의 CUDA 및 관련 라이브러리 사용 (cuDNN, cuBLAS, etc.).
2. 다른 하드웨어 및 소프트웨어 환경과 호환되어야 하는 빌드 대신 최신 소프트웨어 사용자 정의가 포함된 대상 시스템을 위해 특별히 만들어진 소스 빌드를 사용합니다.
3. 최신 버전의 PyTorch 및 보조 라이브러리 사용. 
많은 사용자가 기존 환경과의 호환성 문제로 인해 PyTorch 버전을 업데이트하지 않습니다.
4. 사용자에게 속도 문제에 대한 해답을 어디서 찾는지를 알려줍니다
   (이것이 가장 중요한 요소일 수 있음).

AMP 및 cuDNN 벤치마킹과 같은 기술과 결합하면 __*동일한 하드웨어에서*__
계산 처리량이 극적으로(예: x10) 증가할 수 있습니다.

프로젝트에서 Docker를 사용하지 않으려는 경우에도,
이 템플릿이 유용할 수 있습니다.

**_빌드에서 생성된 휠 파일은 Docker에 의존하지 않고 모든 Python 환경에서 사용할 수 있습니다._**

따라서 이 프로젝트를 사용하여 원하는 환경(`conda`, `pip` 등)에 대해 학습 및 추론 속도를 극적으로 개선하여 
사용자 지정 휠 파일을 생성할 수 있습니다.


## Quickstart
__*사용자는 `Dockerfile`의 `train` 단계를 원하는 대로 자유롭게 사용자 지정할 수 있습니다. 
그러나 절대적으로 필요한 경우가 아니면 `build` 단계를 변경하지 마세요.
새 패키지를 빌드해야 하는 경우, 새 `build` 계층을 추가하세요.*__

이 프로젝트는 템플릿이며, 사용자는 필요에 맞게 사용자 정의해야 합니다.

코드는 필요한 NVIDIA 드라이버와 최신 버전의 Docker 및 Docker Compose가 사전 설치된 Linux 호스트에서 실행되는 것으로 가정합니다.
그렇지 않은 경우, 먼저 설치하세요.

학습 이미지를 빌드하려면,
먼저 `apt`/`conda`/`pip`에서 원하는 패키지를 포함하도록 Dockerfile `train` 단계를 편집합니다.

그런 다음 https://developer.nvidia.com/cuda-gpus 를 방문하여
대상 GPU 장치의 컴퓨팅 기능(CC)을 찾으세요.

마지막으로, `make all CC=TARGET_CC(s)`를 실행하세요.


### Examples 
(1) RTX 3090용 `make all CC="8.6"`, 
(2) RTX 2080Ti 와 RTX 3090용 `make all CC="7.5 8.6"`
(많은 GPU CC용으로 빌드하면 빌드 시간이 늘어남).

그러면 학습에 사용할 수 있는 `pytorch_source:train` 이미지가 생성됩니다.

빌드 중에 사용할 수 없는 장치용 CC는 이미지를 빌드하는 데 사용할 수 있다는것을 주의하세요.
예를 들어, 이미지를 RTX 2080Ti 시스템에서 사용해야 하지만 사용자에게 RTX 3090만 있는 경우, 
사용자는 이미지가 RTX 2080Ti GPU에서 작동하도록 `CC="7.5"`를 설정할 수 있습니다.
`Makefile`에서 `CC`로 지정되는 `TORCH_CUDA_ARCH_LIST`를 설정하는 방법에 대한 자세한 가이드는
https://pytorch.org/docs/stable/cpp_extension.html 를 참조하세요.


### Makefile Explanation
`Makefile` 은 이 패키지를 간단하고 모듈화하여 사용할 수 있도록 설계되었습니다.

생성될 첫 번째 이미지는 빌드에 필요한 모든 패키지가 포함된 `pytorch_source:build_install`입니다.
설치 이미지는 다운로드를 캐시하기 위해 별도로 생성됩니다.

두 번째 이미지는 `pytorch_source:build_torch-v1.9.1`(기본값)입니다.
여기에는 Python 3.8, CUDA 11.3.1 그리고 cuDNN 8이 포함된 
Ubuntu 20.04 LTS의 PyTorch 1.9.1 설정과 함께 PyTorch, TorchVision, TorchText 및 TorchAudio용 휠이 포함되어 있습니다.
두 번째 이미지는 빌드 프로세스의 결과를 캐시하기 위해 존재합니다.

Docker를 사용하지 않고 환경에 pip 설치를 위한 `.whl` 휠 파일만 추출하려는 경우,
생성된 휠 파일은 `/tmp/dist` 디렉토리에서 찾을 수 있습니다.

빌드 결과를 저장하면 다른 PyTorch 버전(다른 CUDA 버전, 다른 라이브러리 버전 등)이 필요한 경우보다 편리한 버전 전환이 가능합니다.

최종 이미지는 실제 학습에 사용할 이미지인 `pytorch_source:train`입니다.
빌드 아티팩트(휠 등)에 대해서만 이전 단계에 의존하고 다른 것은 전혀 사용하지 않습니다.
이를 통해 다양한 환경 및 GPU 장치에 최적화된 다양한 훈련 이미지를 매우 간단하게 생성할 수 있습니다.

PyTorch가 이미 빌드되었기 때문에,
학습 이미지는 나머지 `apt`/`conda`/`pip` 패키지만 다운로드하면 됩니다.
이 프로세스의 속도를 높이기 위해 캐싱도 구현됩니다.


### Timezone Settings
해외 사용자는 이 섹션이 도움이 될 수 있습니다.

`train` 이미지에는 `tzdata` 패키지를 사용하여 `TZ` 변수에 의해 설정된 시간대가 있습니다.
기본 시간대는 `Asia/Seoul` 이지만 `make` 호출 시 `TZ` 변수를 지정하여 변경할 수 있습니다.
[IANA](https://www.iana.org/time-zones) 시간대 이름을 사용하여 원하는 시간대를 지정합니다.

예시: `make all CC="8.6" TZ=America/Los_Angeles` 는 학습 이미지에서 L.A.시간을 사용합니다.

참고: 학습 이미지에만 시간대 설정이 있습니다.
설치 및 이미지 빌드는 시간대 정보를 사용하지 않습니다.

또한, 학습 이미지에는 한국어 사용자를 위해 업데이트된 'apt' 및 'pip' 설치 URL이 있습니다.
설치 캐시로 인해 불필요할 수 있지만,
설치 속도를 높이려면,
해당 위치에 최적화된 URL을 찾으십시오.


## Specific PyTorch Version
__*PyTorch 보조 라이브러리는 일치하는 PyTorch 버전에서만 작동합니다.*__

파이토치 버전을 바꾸기 위해,
[`PYTORCH_VERSION_TAG`](https://github.com/pytorch/pytorch), 
[`TORCHVISION_VERSION_TAG`](https://github.com/pytorch/vision), 
[`TORCHTEXT_VERSION_TAG`](https://github.com/pytorch/text), 및
[`TORCHAUDIO_VERSION_TAG`](https://github.com/pytorch/audio) 
를 변수에 일치하는 버전으로 설정해야합니다.

`*_TAG` 변수는 해당 저장소의 GitHub 태그 또는 브랜치 이름이어야 합니다.
각 라이브러리의 GitHub 레포지토리를 방문하여 적절한 태그를 찾으세요.

예시: PyTorch 1.9.1로 RTX 3090 GPU를 구축하려면 다음 명령을 사용하세요.

`make all CC="8.6" 
PYTORCH_VERSION_TAG=v1.9.1 
TORCHVISION_VERSION_TAG=v0.10.1 
TORCHTEXT_VERSION_TAG=v0.10.1
TORCHAUDIO_VERSION_TAG=v0.9.1`.

결과 이미지 `pytorch_source:train`은 Compute Capability 8.6이 있는 GPU에서 PyTorch 1.9.1로 학습하는 데 사용할 수 있습니다.


## Multiple Training Images
동일한 호스트에서 여러 학습 이미지를 사용하려면 `TRAIN_NAME`에 다른 이름을 지정하세요.
기본값은 `train`입니다.

동일한 빌드 이미지가 다른 학습 이미지에 사용되는 경우 PyTorch를 다시 빌드하지 않고도 새 학습 이미지를 생성할 수 있습니다.
새 학습 이미지를 만드는 데는 기껏해야 몇 분 밖에 걸리지 않습니다.

다음 사용 사례에 유용합니다.
1. 다른 UID/GID를 가진 다른 사용자가 별도의 학습 이미지를 사용하도록 허용할 때
2. 다른 라이브러리 설치 및 구성으로 최종 학습 이미지의 다른 버전을 사용할 때
3. 각기 다른 라이브러리와 설정이 있는 여러 PyTorch 프로젝트에 이 템플릿을 사용할 때

예를 들어 `pytorch_source:build_torch-v1.9.1`이 이미 빌드된 경우,
Alice와 Bob은 다음 명령을 사용하여 별도의 이미지를 만듭니다.

Alice:
`make build-train 
CC="8.6"
TORCH_NAME=build_torch-v1.9.1
PYTORCH_VERSION_TAG=v1.9.1
TORCHVISION_VERSION_TAG=v0.10.1
TORCHTEXT_VERSION_TAG=v0.10.1
TORCHAUDIO_VERSION_TAG=v0.9.1
TRAIN_NAME=train_alice`

Bob:
`make build-train 
CC="8.6"
TORCH_NAME=build_torch-v1.9.1
PYTORCH_VERSION_TAG=v1.9.1
TORCHVISION_VERSION_TAG=v0.10.1
TORCHTEXT_VERSION_TAG=v0.10.1
TORCHAUDIO_VERSION_TAG=v0.9.1
TRAIN_NAME=train_bob` 

이런 식으로 Alice의 이미지에는 그녀의 UID/GID가 있고 Bob의 이미지에는 그의 UID/GID가 있습니다.
학습 이미지에는 빌드 중에 사용자가 설정되어 있기 때문에 이 절차가 필요합니다.
또한. 다른 사용자는 학습 이미지에 다른 라이브러리를 설치할 수 있습니다.
환경 변수 및 기타 설정도 다를 수도 있습니다.


### Word of Caution
 새 학습 이미지를 생성하기 위한 빌드 캐시로써 `pytorch_source:build_torch-v1.9.1`과 같은 빌드 이미지를 사용할때, 
사용자는 모든 이전 레이어의 모든 빌드 인수(--build-arg를 사용하여 ARG 및 ENV에서 지정한 변수)를 다시 지정해야 합니다.

그렇지 않으면 이러한 인수의 기본값이 `Dockerfile`에 제공되고 다른 입력 값으로 인해 캐시 누락이 발생합니다.

이는 이전 레이어를 재구축하는 데 시간을 낭비하고,
더 중요하게는 환경 불일치로 인해 학습 이미지의 불일치를 유발합니다.

여기에는 `docker-compose.yaml` 파일도 포함됩니다.
빌드 중에 `Dockerfile`에 제공된 모든 인수를 다시 지정해야 합니다.
여기에는 `Makefile`에는 있지만 버전 태그와 같이 `Dockerfile`에는 없는 기본값이 포함됩니다.

__*Docker가 이미 빌드한 레이어를 다시 빌드하기 시작하면 빌드 인수가 잘못 지정되었는지 의심하세요.*__ 

자세한 내용은 https://docs.docker.com/develop/develop-images/dockerfile_best-practices/#leverage-build-cache
를 참고하세요.

나중에 캐시로 사용하려면 `BUILDKIT_INLINE_CACHE`도 이미지에 부여해야 합니다. 
자세한 내용은 https://docs.docker.com/engine/reference/commandline/build/#specifying-external-cache-sources
를 참고하세요.


## Advanced Usage
`Makefile`은 고급 사용을 위한 `*-full` 명령을 제공합니다.

`make all-full CC=YOUR_GPU_CC TRAIN_NAME=train_cu102` 는 기본적으로 
`pytorch_source:build_install-ubuntu18.04-cuda10.2-cudnn8-py3.9`,
`pytorch_source:build_torch-v1.9.1-ubuntu18.04-cuda10.2-cudnn8-py3.9`, 
밑 `pytorch_source:train_cu102` 를 생성할 것입니다.

이 이미지는 GTX 1080Ti와 같은 CUDA 10 장치의 교육/배포에 사용할 수 있습니다.

또한, `*-clean` 명령은 이전 빌드에 대한 캐시 의존도를 확인하기 위해 제공됩니다.


### Specific CUDA Version
CUDA 버전을 변경하려면 `CUDA_VERSION`, `CUDNN_VERSION`, `MAGMA_VERSION`을 설정하세요.
필요한 경우 `PYTHON_VERSION`도 변경할 수 있습니다.

그러면 `build-train` 명령으로 학습 이미지를 생성하기 위한 캐시로 사용할 수 있는 빌드 이미지가 생성됩니다.

또한, 프로젝트에서 캐싱을 광범위하게 사용한다는 것은 두 번째 빌드가 첫 번째 빌드보다 훨씬 빠릅니다.
이는 여러 PyTorch/CUDA 버전에 대해 많은 이미지를 생성해야 하는 경우에 유리할 수 있습니다.

### Specific Linux Distro
CentOS 및 UBI 이미지는 `Dockerfile`을 약간만 수정하면 만들 수 있습니다.
전체 지침은 `Dockerfile`을 참고하세요.

나중에 `LINUX_DISTRO` 및 `DISTRO_VERSION` 인수를 설정합니다.

### Windows
Windows 사용자는 Windows 11로 업데이트하고,
Linux용 Windows 하위 시스템(WSL)을 설치하여 이 템플릿을 사용할 수 있습니다.
Windows 11의 WSL은 기본 Linux를 사용하는 것과 유사한 경험을 제공합니다.

이 프로젝트는 WSL CUDA 드라이버 및 Windows용 Docker Desktop을 사용하여 Windows 11의 WSL에서 테스트되었습니다.


# Interactive Development with Docker Compose

## _Raison D'être_
이 섹션의 목적은 딥러닝 개발을 위한 새로운 패러다임을 소개하는 것입니다.
딥 러닝 프로젝트에 Docker Compose를 사용하는 것이
결국 모범 사례가 되어 ML 실험의 재현성을 높이고,
일반 연구원이 개발 환경을 관리해야 하는 부담에서 해방되기를 바랍니다.

딥 러닝 커뮤니티에서는 `conda` 또는 `pip`를 사용하여 로컬 환경에서 개발하는 것이 일반적입니다.
그러나 이것은 개발 환경과 그 환경에서 실행될 코드를 재현할 수 없게 만들 위험이 있습니다.
이것은 이 기사의 많은 독자들이 직접 경험하게 될 과학적 진보에 대한 심각한 해악입니다.

Docker 컨테이너는 다양한 컴퓨팅 환경에서 재현 가능한 프로그램을 제공하기 위한 표준 방법입니다.
호스트 또는 서로의 간섭 없이 프로그램을 실행할 수 있는 격리된 환경을 만듭니다.
자세한것은 https://www.docker.com/resources/what-container 참고하세요.

그러나 실제로, Docker 컨테이너는 종종 오용됩니다.
컨테이너는 일시적이며 모범 사례에 따르면 각 실행에 대해 새 컨테이너를 만들어야 합니다.
그러나 이것은 개발, 
특히 새로운 라이브러리를 지속적으로 설치해야 하고,
버그가 종종 런타임 시에만 나타나는 딥 러닝 애플리케이션의 경우 매우 불편합니다.
이것은 많은 연구자들이 대화형 컨테이너 내부에서 개발하도록 이끕니다.
Docker 사용자는 종종 
`docker run -v my_data:/mnt/data -p 8080:22 -t my_container my_image:latest /bin/bash`와 같은 
명령이 포함된 `run.sh` 파일을 가지고 있으며, SSH를 사용하여 실행 중인 컨테이너에 연결합니다.
VSCode는 컨테이너 내부에서 코딩하는 데 사용할 수 있는 원격 개발 모드도 제공합니다.

이 접근 방식의 문제는 이러한 대화형 컨테이너가 로컬 개발 환경만큼 재현할 수 없게 된다는 것입니다.
실행 중인 컨테이너는 새 포트에 연결하거나 새 볼륨을 연결할 수 없습니다.
그러나 컨테이너 내의 컴퓨팅 환경이 설치 및 빌드의 몇 개월에 걸쳐 생성된 경우,
이를 유지하는 유일한 방법은 이미지로 저장하고,
저장된 이미지에서 새 컨테이너를 생성하는 것입니다.
이 프로세스를 몇 번 반복하면 결과 이미지가 부풀어 오르고 완전히 재현할 수 없게 됩니다.
이 문제를 완화하기 위해 컨테이너를 쉽게 관리할 수 있도록 `docker-compose.yaml` 파일을 제공합니다.
Docker Compose는 Docker의 일부이며 이미 개발 및 프로덕션 모두에서 널리 사용되는 도구입니다.
알 수 없는 이유로, 아직 딥 러닝 커뮤니티에서 시작되지는 않았지만,
Docker 사용 방법을 아는 사람이라면 Compose가 상당히 간단하다는 것을 알게 될 것입니다.
이는 Compose가 단일 컨테이너 개발에도 사용될 수 있지만,
종종 다중 컨테이너 솔루션으로 알려지기 때문일 수 있습니다.

`docker-compose.yaml` 파일을 사용하면 사용자가 빌드 및 실행에 대한 설정을 지정할 수 있습니다.
새 볼륨을 연결하는 것은 현재 컨테이너를 제거하고,
`docker-compose.yaml`/`Dockerfile` 파일에 한 줄을 추가한 다음,
동일한 이미지에서 새 컨테이너를 만드는 것처럼 간단합니다.
빌드 캐시를 사용하면 새 이미지를 매우 빠르게 빌드할 수 있으므로,
Docker 채택에 대한 또 다른 장벽인 긴 초기 빌드 시간이 제거됩니다.

아래 지침은 터미널에서 대화형 개발을 허용하므로,
로컬 개발에서 Docker 및 Docker Compose로 훨씬 더 원활하게 전환할 수 있습니다.

운 좋게도 딥러닝 커뮤니티는 이 기술을 사용하여
"_write once, train anywhere_"할 수 있습니다.
하지만 내가 다수의 사람들에게 나의 방법의 장점을 설명하는것에 실패하더라도, 
`conda` 환경을 조직하는 헛된 노동으로부터, 불행한 대학원생들의 서류 제출이 끝나기 직전에 부수고 태워버리면서, 살려줄수 있습니다.
그들의 서류 제출이 끝나기 직전에 부수고 태워버리면서,


## Usage
__*`Makefile`로 생성된 Docker 이미지는 `docker-compose.yaml` 파일과 완벽하게 호환됩니다.
Docker Compose를 사용하기 위해 지울 필요가 없습니다.*__

Docker Compose V2(https://docs.docker.com/compose/cli-command 참고)를 사용하여 다음 두 명령을 실행합니다.
여기서 `train`은 제공된 `docker-compose.yaml` 파일의 기본 서비스 이름입니다.

0. `docker-compose.yaml`을 읽고 `.env` 파일에 변수를 설정합니다(처음에만 해당).
1. `docker compose up -d train`
2. `docker compose exec train /bin/bash`

그러면 `docker-compose.yaml` 파일의 `train` 서비스에서 지정한 설정으로 대화형 셸이 열립니다.
환경 변수는 프로젝트 루트에 있는 `.env` 파일에 저장할 수 있으므로,
실행할 때마다 UID/GID 값과 같은 변수를 입력할 필요가 없습니다.
기본 `.env` 파일을 생성하려면 `make env`를 실행하세요.

이는 재현 가능한 개발 환경을 관리하는 데 매우 편리합니다.
예를 들어 프로젝트에 대해 새 `pip` 또는 `apt` 패키지를 설치해야 하는 경우, 
사용자는 `apt-get install` 또는 `pip install` 명령에 패키지를 추가하여,
`Dockerfile`의 `train` 레이어를 간단히 편집한 후 다음 명령을 실행할 수 있습니다.

`docker compose up -d --build train`.

이것은 현재 `train` 세션을 제거하고, 이미지를 다시 빌드하고, 새로운 `train` 세션을 시작합니다.
그러나 PyTorch를 다시 빌드하지 않습니다(캐시 누락이 발생하지 않는다고 가정).
따라서 사용자는 캐싱 및 빠른 미러 URL로 가속화되는 추가 다운로드를 위해 몇 분 정도만 기다리면 됩니다.

`Dockerfile` 또는 `docker-compose.yaml` 파일을 편집한 후 서비스를 중지했다가,
다시 시작하려면 `docker compose up -d train`을 다시 실행하기만 하면 됩니다.

모든 Compose 컨테이너를 제거하려면 다음을 사용하세요.

`docker compose down`.
원격 서버가 있는 사용자는 Docker 컨텍스트
(https://docs.docker.com/engine/context/working-with-contexts 참조)
를 사용하여 로컬 환경에서 컨테이너에 액세스할 수 있습니다.
Docker Compose에 대한 자세한 내용은 설명서를 참조하세요.
https://github.com/compose-spec/compose-spec/blob/master/spec.md.


## Compose as Best Practice

이런 방식으로 Docker Compose를 사용하는 것은 이 프로젝트에 의존하지 않는 범용 기술이라는 점을 강조하고 싶습니다.
예를 들어 NVIDIA NGC PyTorch 레포지토리의 이미지가 `ngc.Dockerfile`의 기본 이미지로 사용되었습니다.
NVIDIA NGC PyTorch 이미지에는 최신 GPU 아키텍처에 대한 많은 최적화가 포함되어 있으며,
사전 설치된 다수의 기계 학습 라이브러리를 제공합니다.
새 프로젝트를 시작하여 종속성이 없는 사람에게는 최신 NGC 이미지를 사용하는 것이 좋습니다.

NGC 이미지를 사용하려면 다음 명령을 사용하세요.

1. `docker compose up -d ngc`
2. `docker compose exec ngc /bin/bash`

이전 `train` 세션과의 유일한 차이점은 세션 이름입니다.


# Known Issues

1. `ssh`로 실행 중인 컨테이너에 연결하면 `ENV`에서 설정한 모든 변수가 제거됩니다.
이는 `sshd`가 새로운 환경을 시작하여 이전의 모든 변수를 지우기 때문입니다.
컨테이너를 입력하기 위해 `docker`/`docker compose`를 사용하는 것이 좋습니다.

2. CUDA 11.4.x 기반 빌드는 2021년 10월 현재 사용할 수 없습니다.
`magma-cuda114`가 아나콘다의 `pytorch` 채널에 출시되지 않았기 때문입니다.
사용자는 `magma-cuda`의 이전 버전으로 빌드를 시도하거나,
`conda-forge`에서 사용 가능한 버전을 시도할 수 있습니다.
'magma'의 소스 빌드는 PR로 환영받을 것입니다.

3. Ubuntu 16.04 build fails. 
이는 Ubuntu 16.04에서 `apt`가 설치한 기본 `git`이 `--jobs` 플래그를 지원하지 않기 때문입니다.
`git-core` PPA를 `apt`에 추가하고 최신 버전의 git을 설치합니다.
또한 PyTorch v1.9+는 Ubuntu 16에서 빌드되지 않습니다.
빌드하려면 버전 태그를 v1.8.2로 낮추세요.
그러나 Xenial Xerus가 이미 EOL에 도달했으므로,
프로젝트는 Ubuntu 16.04 빌드를 수용하도록 수정되지 않습니다.


# Desiderata

0. **MORE STARS**. 이 글을 읽고 있다면 즉시 이 저장소에 별표를 표시하세요. 난 진지합니다.

1. CentOS 및 UBI 이미지는 아직 구현되지 않았습니다.
간단한 수정만 필요하므로 이를 구현하는 PR 매우 환영할 것입니다.

2. 다른 언어로의 번역을 환영합니다.
별도의 'LANG.README.md' 파일을 만들어 PR을 생성해 주세요.

3. 이 프로젝트를 자유롭게 공유하십시오! 나는 당신에게 행운과 행복한 코딩을 기원합니다!

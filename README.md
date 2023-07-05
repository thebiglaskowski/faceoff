# Faceoff 

**Img2Image - Img2GIF - Img2MP4**

Swap faces from a source image to a destination medium. Each app runs independelty in it's own Gradio instance for ease of use.

Read more at [https://thebiglaskowski.com/posts/face-swapping-with-ai](https://thebiglaskowski.com/posts/face-swapping-with-ai/)

## Demo: Faceoff Img2MP4 vs DeepFaceLab Deepfake

Faceoff on the left, DeepFaceLab on the right

<video src='https://youtu.be/J51ipLvmScE' width=853/>

## Installation

### Dependencies

1. FFmpeg

From a PowerShell Terminal:

```powershell
winget install Gyan.FFmpeg
```

1. Requirements

```powershell
git clone https://github.com/thebiglaskowski/faceoff.git
cd faceoff
conda create -n faceoff python=3.8 -y
conda activate faceoff
pip install torch==1.11.0+cu115 torchvision==0.12.0+cu115 -f https://download.pytorch.org/whl/torch_stable.html

# If not using CodeFormer enhancement use this torch & torchvision version instead

pip install torch==2.0.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html torchvision==0.15.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html

pip install -r requirements.txt
```

3. Download the [inswapper_128.onnx model](https://huggingface.co/thebiglaskowski/inswapper_128.onnx/tree/main) from Hugging Face and place it in the root directory of this project.

## Usage

### Faceoff Img2Img

```powershell
python faceoff.py
```

<http://127.0.0.1:5000/>

![](assets/arnold.jpg)
![](assets/arnold-faceoff.png)

![](assets/elon.webp)
![](assets/elon-faceoff.png)

### Faceoff Img2GIF

```powershell
python faceoff_gif.py 
```

<http://127.0.0.1:5001/>

![](assets/nacho-libre.gif)
![](assets/nacho-libre-faceoff.gif)

![](assets/smell-rock.gif)
![](assets/smell-rock-faceoff.gif)

### Faceoff Img2MP4

```powershell
python faceoff_video.py
```

<http://127.0.0.1:5002/>

## Demo: Faceoff Img2MP4 - Basic vs Enhanced (w/CodeFormer)

Basic on the left, Enhanced on the right

<video src='https://youtu.be/WLWk4fy5s9w' width=853/>

Sarah Connor?

<video src='https://youtu.be/H7KS8ZoulGw' width=853/>

## To Do

- [ ] Add Enhanced version of Img2GIF
- [ ] Add Enhanced version of Img2MP4

## Special Thanks

- FFMpeg
- InsightFace
- Real-ESRGAN
- CodeFormer
- Open-AI GPT
- PyTorch
- Torchvision

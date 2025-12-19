# Directory **`models/`**
In order for this project to work, you have to download a LLM model in `.gguf` format. 
I recommend to download them form the website:
```
https://huggingface.co/models
```
You have to know how much VRAM/RAM you're working with, and based on that information you can choose a model size. For example if your GPU has 8GB of VRAM, then you should choose a model which is smaller than 7/8 GB. If a chosen model will be bigger than your VRAM capacity, then it will 'spill' in to the system RAM. This will let you run this model at all, but will make your experience a bit slower. Bigger model is let's say 'smarter' than a smaller one, but it is also slower.

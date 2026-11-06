
(Website Link)[https://civitai.red/models/1361228/wan-pov-doggy-style-i2v?modelVersionId=1545040]
Trained for image2video generation, not tested on text2vid

v1.1 UPDATE

Version 1.1 is a significant improvement of v1.0. It produces the doggy style motion consistently. The LoRA was trained on vertical videos - it has been reported that wide aspect videos may have some issues. Please leave some feedback if you have issues with it.

Note: I noticed there can be some image artifacts that appear, usually on the ass. I think this is due to low resolution videos in my data set. I looked into improving the quality with AI upscaling, but ran into a lot off issues that ended up making the videos worse, so this will do for now.

Trigger word is POVdog. "A POV video showing a man having sex doggy style sex with a woman." and "Ass movement and bounce is emphasized" can help too.

I used aipinups69's https://civitai.com/models/1358184/wan-21-penis-cock-dick-lora-t2v-i2v?modelVersionId=1534254 at strength 0.5 in my testing again and it worked great.

v1 -

This is my first attempt at training anything and is still a work in progress. It seems to work okay - some feedback on how to improve is always appreciated. I trained on 480p 14B, I am not sure if it works for the 720p version or not.

Trigger word is: POVdog. The phrase "A POV video showing man having sex doggy style sex with a woman." Was included in most of the training data as well.


In my testing I found this works better with aipinups69's https://civitai.com/models/1358184/wan-21-penis-cock-dick-lora-t2v-i2v?modelVersionId=1534254 at low strength (0.4 - 0.5) to keep the man bits not looking too deformed.

I need to take a closer look to my training process and data set and will hopefully improve this in the week. Training took about 10 hours on two 4090s which seems a lot longer than other people have trained their models, so I definitely have some inefficiencies.

Let me know what y'all think!
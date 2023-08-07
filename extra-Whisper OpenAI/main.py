import os       ## SPEECH TO TEXT USING WHISPER OPENAI
import time
import whisper

print("started")

model = whisper.load_model("large")   #MODELS:  Size	Parameters	English-only model	Multilingual model	Required VRAM	Relative speed
                                                #tiny	    39 M	tiny.en	                    tiny	        ~1 GB	        ~32x
                                                #base	    74 M	base.en	                    base	        ~1 GB	        ~16x
                                                #small	    244 M	small.en	                small	        ~2 GB	        ~6x
                                                #medium	    769 M	medium.en	                medium	        ~5 GB	        ~2x
                                                #large	    1550 M	N/A	                        large	        ~10 GB	         1x

result = model.transcribe("C:/Users/musta/Downloads/output_1.mp3")  ## ADDRESS OF THE AUDIO FILE  

print(result["text"])
print("done")



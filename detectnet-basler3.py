from pypylon import pylon
import cv2
import jetson.utils
import jetson.inference
# conecting to the first available camera
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.Open()

camera.Width.SetValue(3840)
camera.Height.SetValue(2160)
# Grabing Continusely (video) with minimal delay
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly) 
converter = pylon.ImageFormatConverter()
# converting to opencv bgr format
converter.OutputPixelFormat = pylon.PixelType_RGB8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

# Start
display = jetson.utils.glDisplay()
net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)
while camera.IsGrabbing():
	grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

	if grabResult.GrabSucceeded():
		# Access to the image
		frame = converter.Convert(grabResult)
		img = frame.GetArray()
		width = grabResult.Width
		height = grabResult.Height
		# Create Alpha channel
		img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
		# Inference
		cuda = jetson.utils.cudaFromNumpy(img)
		detections = net.Detect(cuda, width, height)
		# Display
		display.RenderOnce(cuda, width, height) 
		# Information
		print("+++++ detected {:d} objects in image".format(len(detections)))
		for detection in detections:
			print(detection)
		net.PrintProfilerTimes()
	grabResult.Release()
# Releasing the resource    
camera.StopGrabbing()
